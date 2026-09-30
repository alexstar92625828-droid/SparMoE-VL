#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cuda:0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/figures/routing_granularity_geometry_preservation/training"

for EXPERT_COUNT in 4 6 8 10; do
  python "${HERE}/train_stage1.py" \
    --expert-count "${EXPERT_COUNT}" \
    --device "${DEVICE}" \
    --output-dir "${OUTPUT_ROOT}/n${EXPERT_COUNT}/stage1"

  python "${HERE}/train_stage2.py" \
    --expert-count "${EXPERT_COUNT}" \
    --stage1-checkpoint "${OUTPUT_ROOT}/n${EXPERT_COUNT}/stage1/best.pt" \
    --device "${DEVICE}" \
    --output-dir "${OUTPUT_ROOT}/n${EXPERT_COUNT}/stage2"
done
