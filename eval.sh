#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${1:?Usage: bash eval.sh <robotwin|libero> <ckpt.pt> <dataset_stats.json> [hydra_overrides...]}"
CKPT="${2:?Checkpoint path is required.}"
STATS="${3:?Dataset stats path is required.}"
shift 3

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${BENCHMARK}" in
  robotwin) exec bash "${PROJECT_ROOT}/scripts/eval_robotwin.sh" "${CKPT}" "${STATS}" "$@" ;;
  libero) exec bash "${PROJECT_ROOT}/scripts/eval_libero.sh" "${CKPT}" "${STATS}" "$@" ;;
  *)
    echo "Unknown benchmark: ${BENCHMARK}. Expected robotwin or libero." >&2
    exit 1
    ;;
esac
