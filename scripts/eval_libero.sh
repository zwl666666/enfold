#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?Usage: bash scripts/eval_libero.sh <ckpt.pt> <dataset_stats.json|EVALUATION.dataset_stats_path=...> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
DATASET_STATS_ARG=()
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  if [[ "${EXTRA_ARGS[0]}" == EVALUATION.dataset_stats_path=* ]]; then
    DATASET_STATS_ARG=("${EXTRA_ARGS[0]}")
    EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
  elif [[ "${EXTRA_ARGS[0]}" != *=* ]]; then
    DATASET_STATS_ARG=("EVALUATION.dataset_stats_path=${EXTRA_ARGS[0]}")
    EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
  fi
fi

if (( ${#DATASET_STATS_ARG[@]} == 0 )); then
  echo "Error: dataset stats is required. Pass either <dataset_stats.json> or EVALUATION.dataset_stats_path=..." >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/libero_egl_env.sh"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export ENFOLD_PYTHON="${ENFOLD_PYTHON:-$(command -v python)}"

TASK_ARG=()
HAS_TASK_OVERRIDE=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == task=* ]]; then
    HAS_TASK_OVERRIDE=1
    break
  fi
done
if (( HAS_TASK_OVERRIDE == 0 )); then
  TASK_ARG=("task=${TASK_CONFIG:-enfold_libero}")
fi

LIBERO_ENV_CREATE_RETRIES=5 LIBERO_ENV_CREATE_RETRY_DELAY_SEC=300 "${ENFOLD_PYTHON}" experiments/libero/run_libero_manager.py \
  --config-name sim_libero \
  "${TASK_ARG[@]}" \
  "ckpt=${CKPT}" \
  "${DATASET_STATS_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
