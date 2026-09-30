#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:-0}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-${repository_root}/outputs/clip_vitl14_text_ffn_baselines/teal}"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"

for seed in 42 123 2026; do
  seed_output="${output_root}/seed_${seed}"
  "${python_bin}" "${script_dir}/calibrate.py" \
    --output-dir "${seed_output}" \
    --seed "${seed}" \
    --device cuda:0 \
    --batch-size 64 \
    --target-ffn-reduction 0.425 \
    --base-step-size 0.05 \
    --histogram-bins 10000

  "${python_bin}" "${script_dir}/evaluate.py" \
    --checkpoint "${seed_output}/teal_thresholds.pt" \
    --output "${seed_output}/result.json" \
    --device cuda:0 \
    --image-batch-size 128 \
    --text-batch-size 512 \
    --workers 8
done

"${python_bin}" "${script_dir}/summarize.py" \
  --root "${output_root}" \
  --output "${output_root}/summary.json"

