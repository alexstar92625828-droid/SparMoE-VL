#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cuda:0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/sparmoe_vl_clip_vitl14/studies/routing_granularity_geometry_preservation"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/figures/routing_granularity_geometry_preservation"

python "${HERE}/evaluate.py" \
  --checkpoint-n4 "${CHECKPOINT_ROOT}/n4/stage2/best.pt" \
  --checkpoint-n6 "${CHECKPOINT_ROOT}/n6/stage2/best.pt" \
  --checkpoint-n8 "${CHECKPOINT_ROOT}/n8/stage2/best.pt" \
  --checkpoint-n10 "${CHECKPOINT_ROOT}/n10/stage2/best.pt" \
  --device "${DEVICE}" \
  --output "${OUTPUT_ROOT}/analysis.json"

python "${HERE}/plot.py" \
  --input "${OUTPUT_ROOT}/analysis.json" \
  --output-stem "${OUTPUT_ROOT}/routing_geometry_preservation"
