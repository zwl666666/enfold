#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: NNODES=<num_nodes> NODE_RANK=<rank> MASTER_ADDR=<rank0_ip> [MASTER_PORT=29500] bash scripts/train_zero1_multinode.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:?Set NNODES to the total number of nodes.}"
MACHINE_RANK="${NODE_RANK:?Set NODE_RANK to this node rank, starting from 0.}"
MAIN_PROCESS_IP="${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node IP or hostname.}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NPROC_PER_NODE}" || ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}), NNODES (${NUM_MACHINES}), and NODE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

TOTAL_PROCESSES="$((NPROC_PER_NODE * NUM_MACHINES))"

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
  RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

  export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
  export RUN_ID_SYNC_PORT
  export RUN_ID_SYNC_TIMEOUT
  export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
  export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
  export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

  RUN_ID="$(
    python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
  )"

  echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_processes=${TOTAL_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT} run_id=${RUN_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --deepspeed_multinode_launcher standard \
  --num_processes "${TOTAL_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
