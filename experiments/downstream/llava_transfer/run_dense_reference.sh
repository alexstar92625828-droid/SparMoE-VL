#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
experiment="experiments/downstream/llava_transfer"
output_root="${OUTPUT_ROOT:-outputs/downstream/llava_transfer/evaluation}"
dense_root="${output_root}/dense"

python3 "${experiment}/evaluate_pope.py" \
  --mode dense --device "${device}" --output-dir "${dense_root}/pope"
python3 "${experiment}/evaluate_mme.py" \
  --mode dense --device "${device}" --output-dir "${dense_root}/mme_p"
python3 "${experiment}/evaluate_gqa.py" \
  --mode dense --device "${device}" --output-dir "${dense_root}/gqa"
python3 "${experiment}/evaluate_vqav2.py" \
  --mode dense --device "${device}" --output-dir "${dense_root}/vqav2"
