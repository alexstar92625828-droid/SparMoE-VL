#!/usr/bin/env bash
set -euo pipefail

physical_gpu="${1:-0}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-${repository_root}/outputs/clip_vitl14_vision_ffn_baselines/optin}"
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"

for setting in "42:0.348" "123:0.354" "2026:0.361"; do
  seed="${setting%%:*}"
  reduction="${setting##*:}"
  seed_output="${output_root}/seed_${seed}"
  shard_output="${seed_output}/shards/shard_000_of_001"
  mkdir -p "${shard_output}"

  if [[ ! -f "${shard_output}/optin_score_shard.pt" ]]; then
    resume_args=()
    if [[ -f "${shard_output}/optin_search_progress.pt" ]]; then
      resume_args+=(--resume)
    fi
    "${python_bin}" "${script_dir}/score.py" \
      --output-dir "${shard_output}" \
      --seed "${seed}" \
      --shard-index 0 \
      --num-shards 1 \
      --device cuda:0 \
      --batch-size 32 \
      --candidate-batch 1 \
      --workers 8 \
      "${resume_args[@]}"
  fi

  if [[ ! -f "${seed_output}/optin_vision_pruned.pt" ]]; then
    "${python_bin}" "${script_dir}/prune.py" \
      --score-root "${seed_output}/shards" \
      --output-dir "${seed_output}" \
      --seed "${seed}" \
      --target-ffn-reduction "${reduction}" \
      --device cuda:0
  fi

  "${python_bin}" "${script_dir}/evaluate.py" \
    --checkpoint "${seed_output}/optin_vision_pruned.pt" \
    --output "${seed_output}/result.json" \
    --device cuda:0 \
    --image-batch-size 32 \
    --text-batch-size 256 \
    --workers 8
done

"${python_bin}" "${script_dir}/summarize.py" \
  --root "${output_root}" \
  --output "${output_root}/summary.json"
