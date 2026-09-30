#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
experiment="experiments/downstream/llava_transfer"
output_root="${OUTPUT_ROOT:-outputs/downstream/llava_transfer/evaluation}"

bash "${experiment}/run_dense_reference.sh" "${device}"
for seed in 42 123 2026; do
  bash "${experiment}/run_sparse_seed.sh" "${seed}" "${device}"
done
python3 "${experiment}/summarize.py" \
  --dense-root "${output_root}/dense" \
  --sparse-root "${output_root}/sparse" \
  --output-dir "${output_root}/summary"
