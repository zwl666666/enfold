#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${1:?Usage: bash train.sh <robotwin|libero> [num_gpus] [hydra_overrides...]}"
NUM_GPUS="${2:-8}"
shift
if (( $# > 0 )); then
  shift
fi

case "${BENCHMARK}" in
  robotwin) TASK=enfold_robotwin ;;
  libero) TASK=enfold_libero ;;
  *)
    echo "Unknown benchmark: ${BENCHMARK}. Expected robotwin or libero." >&2
    exit 1
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/scripts/train_zero1.sh" "${NUM_GPUS}" "task=${TASK}" "$@"
