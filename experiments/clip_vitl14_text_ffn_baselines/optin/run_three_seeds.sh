#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:-0}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-${repository_root}/outputs/clip_vitl14_text_ffn_baselines/optin}"

export CUDA_VISIBLE_DEVICES="${physical_gpu}"

"${python_bin}" "${script_dir}/prepare_data.py" \
  --device cuda:0 \
  --image-batch-size 128 \
  --workers 8

for setting in "42:0.441" "123:0.450" "2026:0.462"; do
  seed="${setting%%:*}"
  reduction="${setting##*:}"
  seed_output="${output_root}/seed_${seed}"
  "${python_bin}" "${script_dir}/prune.py" \
    --output-dir "${seed_output}" \
    --seed "${seed}" \
    --target-ffn-reduction "${reduction}" \
    --device cuda:0 \
    --batch-size 32 \
    --save-every 64 \
    --log-every 256 \
    --resume

  "${python_bin}" "${script_dir}/evaluate.py" \
    --checkpoint "${seed_output}/optin_text_pruned.pt" \
    --output "${seed_output}/result.json" \
    --device cuda:0 \
    --image-batch-size 128 \
    --text-batch-size 512 \
    --workers 8
done

"${python_bin}" "${script_dir}/summarize.py" \
  --root "${output_root}" \
  --output "${output_root}/summary.json"

