#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cuda:0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
RESEARCH_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"
LAYERWISE_ANALYSIS="${LAYERWISE_ANALYSIS:-${PROJECT_ROOT}/outputs/figures/layerwise_representation_consistency/analysis.json}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/figures/cross_modal_similarity_preservation"
PRETRAINED="${PRETRAINED:-${RESEARCH_ROOT}/models/ViT-L-14.pt}"
COCO_ANNOTATIONS="${COCO_ANNOTATIONS:-${RESEARCH_ROOT}/data/eval/coco/annotations/captions_val2017.json}"
COCO_IMAGES="${COCO_IMAGES:-${RESEARCH_ROOT}/data/eval/coco/val2017}"

python "${HERE}/analyze.py" \
  --layerwise-analysis "${LAYERWISE_ANALYSIS}" \
  --pretrained "${PRETRAINED}" \
  --coco-annotations "${COCO_ANNOTATIONS}" \
  --coco-images "${COCO_IMAGES}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_ROOT}"

python "${HERE}/plot.py" \
  --input "${OUTPUT_ROOT}/analysis.json" \
  --coco-annotations "${COCO_ANNOTATIONS}" \
  --coco-images "${COCO_IMAGES}" \
  --output-dir "${OUTPUT_ROOT}"
