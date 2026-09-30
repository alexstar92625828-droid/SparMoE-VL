#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"
study_dir="experiments/sparmoe_vl_clip_vitl14/studies/vision_budget_sweep"
output_root="${OUTPUT_ROOT:-outputs/studies/vision_budget_sweep}"
main_vision_root="${MAIN_VISION_ROOT:-outputs/main/vision}"

for target in 0.4 0.5 0.6 0.7 0.8; do
  tag="p0${target#0.}"
  for seed in 42 123 2026; do
    run_dir="${output_root}/${tag}/seed_${seed}"
    if [[ "${target}" == "0.7" ]]; then
      checkpoint="${main_vision_root}/seed_${seed}/stage2/best.pt"
      if [[ ! -s "${checkpoint}" ]]; then
        echo "p=0.7 reuses the visual main checkpoint; missing ${checkpoint}" >&2
        exit 1
      fi
    else
      python3 "${study_dir}/train_stage1.py" \
        --device "${device}" \
        --target-ratio "${target}" \
        --seed "${seed}" \
        --data-seed 42 \
        --output-dir "${run_dir}/stage1"
      python3 "${study_dir}/train_stage2.py" \
        --device "${device}" \
        --target-ratio "${target}" \
        --seed "${seed}" \
        --data-seed 42 \
        --stage1-checkpoint "${run_dir}/stage1/best.pt" \
        --output-dir "${run_dir}/stage2"
      checkpoint="${run_dir}/stage2/best.pt"
    fi
    python3 "${study_dir}/evaluate.py" \
      --device "${device}" \
      --checkpoint "${checkpoint}" \
      --output "${run_dir}/evaluation.json"
  done
done

python3 "${study_dir}/summarize.py" \
  --results-root "${output_root}" \
  --output-dir "${output_root}/summary"
python3 "${study_dir}/plot_pareto.py" \
  --summary "${output_root}/summary/summary.json" \
  --output "${output_root}/summary/pareto.png"
