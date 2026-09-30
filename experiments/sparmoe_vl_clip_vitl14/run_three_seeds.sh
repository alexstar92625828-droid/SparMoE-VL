#!/usr/bin/env bash
set -euo pipefail

modality="${1:?usage: run_three_seeds.sh vision|text [device]}"
device="${2:-cuda:0}"
case "${modality}" in
  vision) target="0.7" ;;
  text) target="0.6" ;;
  *) echo "modality must be vision or text" >&2; exit 2 ;;
esac

for seed in 42 123 2026; do
  run_root="outputs/main/${modality}/seed_${seed}"
  python3 "experiments/sparmoe_vl_clip_vitl14/${modality}/train_stage1.py" \
    --device "${device}" --seed "${seed}" --data-seed 42 \
    --target-ratio "${target}" --output-dir "${run_root}/stage1"
  python3 "experiments/sparmoe_vl_clip_vitl14/${modality}/train_stage2.py" \
    --device "${device}" --seed "${seed}" --data-seed 42 \
    --target-ratio "${target}" \
    --stage1-checkpoint "${run_root}/stage1/best.pt" \
    --output-dir "${run_root}/stage2"
done
