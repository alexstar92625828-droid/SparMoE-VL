#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
study_dir="experiments/sparmoe_vl_clip_vitl14/studies/cross_dataset_generalization"
output_root="${OUTPUT_ROOT:-outputs/studies/cross_dataset_generalization}"
vision_root="${MAIN_VISION_ROOT:-outputs/main/vision}"
text_root="${MAIN_TEXT_ROOT:-outputs/main/text}"
cache="${output_root}/dense_cache.pt"

if [[ ! -s "${cache}" ]]; then
  python3 "${study_dir}/prepare_dense_cache.py" \
    --device "${device}" \
    --output "${cache}"
fi

for seed in 42 123 2026; do
  vision_checkpoint="${vision_root}/seed_${seed}/stage2/best.pt"
  text_checkpoint="${text_root}/seed_${seed}/stage2/best.pt"
  if [[ ! -s "${vision_checkpoint}" ]]; then
    echo "missing visual main checkpoint: ${vision_checkpoint}" >&2
    exit 1
  fi
  if [[ ! -s "${text_checkpoint}" ]]; then
    echo "missing text main checkpoint: ${text_checkpoint}" >&2
    exit 1
  fi
  python3 "${study_dir}/evaluate_vision.py" \
    --device "${device}" \
    --checkpoint "${vision_checkpoint}" \
    --cache "${cache}" \
    --output "${output_root}/vision/seed_${seed}/evaluation.json"
  python3 "${study_dir}/evaluate_text.py" \
    --device "${device}" \
    --checkpoint "${text_checkpoint}" \
    --cache "${cache}" \
    --output "${output_root}/text/seed_${seed}/evaluation.json"
done

python3 "${study_dir}/summarize.py" \
  --results-root "${output_root}" \
  --output-dir "${output_root}/summary"
