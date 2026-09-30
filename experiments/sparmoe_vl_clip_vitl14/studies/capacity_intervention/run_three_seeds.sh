#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cuda:0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/sparmoe_vl_clip_vitl14/studies/capacity_intervention"

for SEED in 42 123 2026; do
  STAGE1_DIR="${OUTPUT_ROOT}/training/seed_${SEED}/stage1"
  STAGE2_DIR="${OUTPUT_ROOT}/training/seed_${SEED}/stage2"
  RESULT="${OUTPUT_ROOT}/evaluation/seed_${SEED}/result.json"
  python "${HERE}/train_stage1.py" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --output-dir "${STAGE1_DIR}"
  python "${HERE}/train_stage2.py" \
    --seed "${SEED}" \
    --stage1-checkpoint "${STAGE1_DIR}/best.pt" \
    --device "${DEVICE}" \
    --output-dir "${STAGE2_DIR}"
  python "${HERE}/evaluate.py" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --checkpoint "${STAGE2_DIR}/best.pt" \
    --output "${RESULT}"
done

python "${HERE}/summarize.py"
