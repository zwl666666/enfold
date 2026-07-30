#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:?Usage: bash scripts/eval_robotwin.sh <ckpt.pt> <dataset_stats.json|EVALUATION.dataset_stats_path=...> [hydra_overrides...]}"
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
  TASK_ARG=("task=${TASK_CONFIG:-enfold_robotwin}")
fi

"${ENFOLD_PYTHON}" experiments/robotwin/run_robotwin_manager.py \
  --config-name sim_robotwin \
  "${TASK_ARG[@]}" \
  "ckpt=${CKPT}" \
  "${DATASET_STATS_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
