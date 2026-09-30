#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:-0}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-${repository_root}/outputs/clip_vitl14_vision_ffn_baselines/flap}"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"

for seed in 42 123 2026; do
  seed_output="${output_root}/seed_${seed}"
  mkdir -p "${seed_output}"
  if [[ ! -f "${seed_output}/flap_vision_pruned.pt" ]]; then
    resume_args=()
    if [[ -f "${seed_output}/flap_calibration_progress.pt" ]]; then
      resume_args+=(--resume)
    fi
    "${python_bin}" "${script_dir}/prune.py" \
      --output-dir "${seed_output}" \
      --seed "${seed}" \
      --target-ffn-reduction 0.3563 \
      --device cuda:0 \
      --batch-size 128 \
      --workers 8 \
      --log-every 20 \
      --save-every 100 \
      "${resume_args[@]}"
  fi

  "${python_bin}" "${script_dir}/evaluate.py" \
    --checkpoint "${seed_output}/flap_vision_pruned.pt" \
    --output "${seed_output}/result.json" \
    --device cuda:0 \
    --image-batch-size 32 \
    --text-batch-size 256 \
    --workers 8
done

"${python_bin}" "${script_dir}/summarize.py" \
  --root "${output_root}" \
  --output "${output_root}/summary.json"
