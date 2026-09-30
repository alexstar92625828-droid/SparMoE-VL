#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED DEVICE" >&2
  exit 2
fi

SEED="$1"
DEVICE="$2"
case "${SEED}" in
  42|123|2026) ;;
  *) echo "seed must be 42, 123, or 2026" >&2; exit 2 ;;
esac

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${PROJECT_ROOT}/outputs/main/vision/seed_${SEED}"
STATUS_FILE="${RUN_ROOT}/pipeline.status"

mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"

mark_failed() {
  local exit_code=$?
  printf 'failed exit_code=%s\n' "${exit_code}" > "${STATUS_FILE}"
  exit "${exit_code}"
}
trap mark_failed ERR

printf 'stage1_running\n' > "${STATUS_FILE}"
if [[ -s "${RUN_ROOT}/stage1/best.pt" ]]; then
  printf 'reusing completed Stage-1 checkpoint: %s\n' \
    "${RUN_ROOT}/stage1/best.pt"
else
  "${PYTHON_BIN}" "${HERE}/train_stage1.py" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --data-seed 42 \
    --target-ratio 0.7 \
    --output-dir "${RUN_ROOT}/stage1"
fi

printf 'stage1_complete_stage2_running\n' > "${STATUS_FILE}"
if [[ -s "${RUN_ROOT}/stage2/best.pt" ]]; then
  printf 'reusing completed Stage-2 checkpoint: %s\n' \
    "${RUN_ROOT}/stage2/best.pt"
else
  "${PYTHON_BIN}" "${HERE}/train_stage2.py" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --data-seed 42 \
    --target-ratio 0.7 \
    --stage1-checkpoint "${RUN_ROOT}/stage1/best.pt" \
    --output-dir "${RUN_ROOT}/stage2"
fi

printf 'stage2_complete_evaluation_running\n' > "${STATUS_FILE}"
"${PYTHON_BIN}" "${HERE}/evaluate.py" \
  --device "${DEVICE}" \
  --checkpoint "${RUN_ROOT}/stage2/best.pt" \
  --output "${RUN_ROOT}/evaluation/coco.json"

printf 'complete\n' > "${STATUS_FILE}"
