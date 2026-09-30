#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 SEED [DEVICE] [CHECKPOINT]" >&2
  exit 2
fi

seed="$1"
device="${2:-cuda:0}"
experiment="experiments/downstream/llava_transfer"
output_root="${OUTPUT_ROOT:-outputs/downstream/llava_transfer/evaluation}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints/sparmoe_vl_clip336_llava}"
checkpoint="${3:-${checkpoint_root}/seed_${seed}/stage2/best.pt}"
run_root="${output_root}/sparse/seed_${seed}"

case "${seed}" in
  42|123|2026) ;;
  *) echo "seed must be one of 42, 123, 2026" >&2; exit 2 ;;
esac

python3 "${experiment}/evaluate_pope.py" \
  --mode sparse --device "${device}" --checkpoint "${checkpoint}" \
  --output-dir "${run_root}/pope"
python3 "${experiment}/evaluate_mme.py" \
  --mode sparse --device "${device}" --checkpoint "${checkpoint}" \
  --output-dir "${run_root}/mme_p"
python3 "${experiment}/evaluate_gqa.py" \
  --mode sparse --device "${device}" --checkpoint "${checkpoint}" \
  --output-dir "${run_root}/gqa"

starts=(0 53588 107176 160764)
ends=(53588 107176 160764 214354)
chunk_dirs=()
for index in 0 1 2 3; do
  chunk="${run_root}/vqav2_chunks/chunk_${index}"
  chunk_dirs+=("${chunk}")
  python3 "${experiment}/evaluate_vqav2.py" \
    --mode sparse --device "${device}" --checkpoint "${checkpoint}" \
    --start-index "${starts[$index]}" --end-index "${ends[$index]}" \
    --output-dir "${chunk}"
done
python3 "${experiment}/merge_vqav2.py" \
  --mode sparse --checkpoint "${checkpoint}" \
  --chunk-dirs "${chunk_dirs[@]}" --output-dir "${run_root}/vqav2"
python3 "${experiment}/measure_macs.py" \
  --device "${device}" --checkpoint "${checkpoint}" \
  --output-dir "${run_root}/macs"
