#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
experiment="experiments/downstream/llava_transfer"
output_root="${OUTPUT_ROOT:-outputs/downstream/llava_transfer/training}"

for seed in 42 123 2026; do
  stage1="${output_root}/seed_${seed}/stage1"
  stage2="${output_root}/seed_${seed}/stage2"
  python3 "${experiment}/train_stage1.py" \
    --seed "${seed}" \
    --device "${device}" \
    --output-dir "${stage1}"
  python3 "${experiment}/train_stage2.py" \
    --seed "${seed}" \
    --device "${device}" \
    --stage1-checkpoint "${stage1}/best.pt" \
    --output-dir "${stage2}"
done
