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
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/figures/layerwise_representation_consistency"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/sparmoe_vl_clip_vitl14/vision/seed_42/stage2/best.pt}"
PRETRAINED="${PRETRAINED:-${RESEARCH_ROOT}/models/ViT-L-14.pt}"
COCO_ANNOTATIONS="${COCO_ANNOTATIONS:-${RESEARCH_ROOT}/data/eval/coco/annotations/captions_val2017.json}"
COCO_IMAGES="${COCO_IMAGES:-${RESEARCH_ROOT}/data/eval/coco/val2017}"

python "${HERE}/analyze.py" \
  --checkpoint "${CHECKPOINT}" \
  --pretrained "${PRETRAINED}" \
  --coco-annotations "${COCO_ANNOTATIONS}" \
  --coco-images "${COCO_IMAGES}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_ROOT}"

python "${HERE}/plot.py" \
  --input "${OUTPUT_ROOT}/analysis.json" \
  --output-dir "${OUTPUT_ROOT}"
