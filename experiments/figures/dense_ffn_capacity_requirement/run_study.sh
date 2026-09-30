#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cuda:0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/figures/dense_ffn_capacity_requirement"
SHARD_ROOT="${OUTPUT_ROOT}/shards"

python "${HERE}/analyze.py" \
  --device "${DEVICE}" \
  --layer-start 1 \
  --layer-end 12 \
  --output-dir "${SHARD_ROOT}"

python "${HERE}/analyze.py" \
  --device "${DEVICE}" \
  --layer-start 13 \
  --layer-end 24 \
  --output-dir "${SHARD_ROOT}"

python "${HERE}/merge.py" \
  "${SHARD_ROOT}/layers_01_12_result.json" \
  "${SHARD_ROOT}/layers_13_24_result.json" \
  --output "${OUTPUT_ROOT}/analysis.json"

python "${HERE}/plot.py" \
  --input "${OUTPUT_ROOT}/analysis.json" \
  --output-stem "${OUTPUT_ROOT}/dense_ffn_capacity_requirement"
