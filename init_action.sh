#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${1:?Usage: bash init_action.sh <robotwin|libero> [initializer_args...]}"
shift

case "${BENCHMARK}" in
  robotwin) TASK=enfold_robotwin ;;
  libero) TASK=enfold_libero ;;
  *)
    echo "Unknown benchmark: ${BENCHMARK}. Expected robotwin or libero." >&2
    exit 1
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
exec "${ENFOLD_PYTHON:-$(command -v python)}" scripts/initialize_action_dit.py \
  --task "${TASK}" \
  --device cuda \
  --dtype bfloat16 \
  "$@"
