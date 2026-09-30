#!/usr/bin/env bash
set -euo pipefail

visible_gpus="${1:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "${visible_gpus}"
if [[ "${#gpu_ids[@]}" -ne 8 ]]; then
  echo "visual MoPE recovery requires exactly 8 comma-separated GPU ids" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-${repository_root}/outputs/clip_vitl14_vision_ffn_baselines/mope}"
selection_dir="${output_root}/structure_selection"
selection_gpu="${gpu_ids[0]}"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${selection_dir}"
CUDA_VISIBLE_DEVICES="${selection_gpu}" "${python_bin}" "${script_dir}/prepare_data.py" \
  --device cuda:0 \
  --output "${selection_dir}/data_manifest.json"

selection_resume=()
if [[ -f "${selection_dir}/taylor_progress.pt" || -f "${selection_dir}/selection_progress.pt" ]]; then
  selection_resume+=(--resume)
fi
if [[ ! -f "${selection_dir}/selection.pt" ]]; then
  CUDA_VISIBLE_DEVICES="${selection_gpu}" "${python_bin}" "${script_dir}/select_structure.py" \
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
    CUDA_VISIBLE_DEVICES="${visible_gpus}" "${python_bin}" -m torch.distributed.run \
      --standalone --nproc_per_node=8 \
      "${script_dir}/train_stage1.py" \
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
    CUDA_VISIBLE_DEVICES="${visible_gpus}" "${python_bin}" -m torch.distributed.run \
      --standalone --nproc_per_node=8 \
      "${script_dir}/train_stage2.py" \
      --seed "${seed}" \
      --selection "${selection_dir}/selection.pt" \
      --stage1-checkpoint "${stage1_dir}/stage1.pt" \
      --output-dir "${stage2_dir}" \
      "${resume_args[@]}"
  fi

  CUDA_VISIBLE_DEVICES="${selection_gpu}" "${python_bin}" "${script_dir}/evaluate.py" \
    --device cuda:0 \
    --selection "${selection_dir}/selection.pt" \
    --checkpoint "${stage2_dir}/stage2.pt" \
    --output "${run_dir}/result.json"
done

"${python_bin}" "${script_dir}/summarize.py" \
  --root "${output_root}" \
  --output "${output_root}/summary.json"
