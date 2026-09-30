#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [device]" >&2
  exit 2
fi

DEVICE="${1:-cuda:0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/sparmoe_vl_clip_vitl14/studies/component_ablation"
MAIN_CHECKPOINT_ROOT="${MAIN_CHECKPOINT_ROOT:-${PROJECT_ROOT}/outputs/main/vision}"

run_trained_method() {
  local method="$1"
  shift
  for seed in "$@"; do
    local checkpoint_root="${OUTPUT_ROOT}/training/${method}/seed_${seed}"
    local stage1_dir="${checkpoint_root}/stage1"
    local stage2_dir="${checkpoint_root}/stage2"
    local result="${OUTPUT_ROOT}/evaluation/${method}/seed_${seed}/result.json"
    python "${HERE}/train_stage1.py" \
      --variant "${method}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --output-dir "${stage1_dir}"
    python "${HERE}/train_stage2.py" \
      --variant "${method}" \
      --seed "${seed}" \
      --stage1-checkpoint "${stage1_dir}/best.pt" \
      --device "${DEVICE}" \
      --output-dir "${stage2_dir}"
    python "${HERE}/evaluate.py" \
      --method "${method}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --checkpoint "${stage2_dir}/best.pt" \
      --output "${result}"
  done
}

run_main_method() {
  local method="$1"
  for seed in 42 123 2026; do
    local checkpoint="${MAIN_CHECKPOINT_ROOT}/seed_${seed}/stage2/best.pt"
    local result="${OUTPUT_ROOT}/evaluation/${method}/seed_${seed}/result.json"
    python "${HERE}/evaluate.py" \
      --method "${method}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --checkpoint "${checkpoint}" \
      --output "${result}"
  done
}

run_trained_method without_spg 42 123 2026
run_trained_method without_layer_adaptive_budget 42 123 2026
run_trained_method without_geometry_preservation 42 123 3407
run_main_method without_token_router
run_main_method sparmoe_vl

python "${HERE}/summarize.py"
