#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:-0}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-${repository_root}/outputs/clip_vitl14_text_ffn_baselines/mope}"
selection_dir="${output_root}/structure_selection"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${selection_dir}"
CUDA_VISIBLE_DEVICES="${physical_gpu}" "${python_bin}" "${script_dir}/prepare_data.py" \
  --device cuda:0 \
  --output "${selection_dir}/data_manifest.json"

selection_resume=()
if [[ -f "${selection_dir}/selection_progress.pt" ]]; then
  selection_resume+=(--resume)
fi
if [[ ! -f "${selection_dir}/selection.pt" ]]; then
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${python_bin}" "${script_dir}/select_structure.py" \
    --device cuda:0 \
    --output-dir "${selection_dir}" \
    "${selection_resume[@]}"
fi

for seed in 42 123 2026; do
  run_dir="${output_root}/seed_${seed}"
  stage1_dir="${run_dir}/stage1"
  stage2_dir="${run_dir}/stage2"
  mkdir -p "${stage1_dir}" "${stage2_dir}"

  if [[ ! -f "${stage1_dir}/stage1.pt" ]]; then
    resume_args=()
    if [[ -f "${stage1_dir}/latest.pt" ]]; then
      resume_args+=(--resume "${stage1_dir}/latest.pt")
    fi
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${python_bin}" "${script_dir}/train_stage1.py" \
      --device cuda:0 \
      --seed "${seed}" \
      --selection "${selection_dir}/selection.pt" \
      --output-dir "${stage1_dir}" \
      "${resume_args[@]}"
  fi

  if [[ ! -f "${stage2_dir}/stage2.pt" ]]; then
    resume_args=()
    if [[ -f "${stage2_dir}/latest.pt" ]]; then
      resume_args+=(--resume "${stage2_dir}/latest.pt")
    fi
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${python_bin}" "${script_dir}/train_stage2.py" \
      --device cuda:0 \
      --seed "${seed}" \
      --selection "${selection_dir}/selection.pt" \
      --stage1-checkpoint "${stage1_dir}/stage1.pt" \
      --output-dir "${stage2_dir}" \
      "${resume_args[@]}"
  fi

  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${python_bin}" "${script_dir}/evaluate.py" \
    --device cuda:0 \
    --selection "${selection_dir}/selection.pt" \
    --checkpoint "${stage2_dir}/stage2.pt" \
    --output "${run_dir}/result.json"
done

"${python_bin}" "${script_dir}/summarize.py" \
  --root "${output_root}" \
  --output "${output_root}/summary.json"
