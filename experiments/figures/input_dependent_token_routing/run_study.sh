#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cpu}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
RESEARCH_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/figures/input_dependent_token_routing"
VISION_CHECKPOINT="${VISION_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/sparmoe_vl_clip_vitl14/vision/seed_42/stage2/best.pt}"
TEXT_CHECKPOINT="${TEXT_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/sparmoe_vl_clip_vitl14/text/seed_42/stage2/best.pt}"
PRETRAINED="${PRETRAINED:-${RESEARCH_ROOT}/models/ViT-L-14.pt}"
IMAGE_ROOT="${IMAGE_ROOT:-${RESEARCH_ROOT}/data/eval/coco/val2017}"

python "${HERE}/analyze.py" \
  --vision-checkpoint "${VISION_CHECKPOINT}" \
  --text-checkpoint "${TEXT_CHECKPOINT}" \
  --pretrained "${PRETRAINED}" \
  --image-root "${IMAGE_ROOT}" \
  --device "${DEVICE}" \
  --output "${OUTPUT_ROOT}/analysis.json"

python "${HERE}/plot.py" \
  --input "${OUTPUT_ROOT}/analysis.json" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
