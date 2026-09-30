#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cpu}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/figures/layerwise_capacity_allocation"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/sparmoe_vl_clip_vitl14/vision/seed_42/stage2/best.pt}"

python "${HERE}/analyze.py" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --output "${OUTPUT_ROOT}/analysis.json"

python "${HERE}/plot.py" \
  --input "${OUTPUT_ROOT}/analysis.json" \
  --output-dir "${OUTPUT_ROOT}"
