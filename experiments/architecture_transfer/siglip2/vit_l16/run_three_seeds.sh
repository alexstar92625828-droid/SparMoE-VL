#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {vision|text} [device]" >&2
  exit 2
fi

MODALITY="$1"
DEVICE="${2:-cuda:0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/architecture_transfer/siglip2/vit_l16/${MODALITY}"

for SEED in 42 123 2026; do
  STAGE1_DIR="${OUTPUT_ROOT}/seed_${SEED}/stage1"
  STAGE2_DIR="${OUTPUT_ROOT}/seed_${SEED}/stage2"
  python "${HERE}/train_stage1.py" \
    --modality "${MODALITY}" --seed "${SEED}" \
    --device "${DEVICE}" --output-dir "${STAGE1_DIR}"
  python "${HERE}/train_stage2.py" \
    --modality "${MODALITY}" --seed "${SEED}" \
    --device "${DEVICE}" --stage1-checkpoint "${STAGE1_DIR}/best.pt" \
    --output-dir "${STAGE2_DIR}"
  python "${HERE}/evaluate.py" \
    --modality "${MODALITY}" --seed "${SEED}" \
    --device "${DEVICE}" --checkpoint "${STAGE2_DIR}/best.pt" \
    --output "${OUTPUT_ROOT}/seed_${SEED}/retrieval.json"
done

python "${HERE}/summarize.py" --modality "${MODALITY}"
