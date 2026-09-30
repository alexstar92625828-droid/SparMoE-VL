#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${HERE}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-512}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/main/vision"
CONTROLLER_ROOT="${OUTPUT_ROOT}/controller"
CONTROLLER_STATUS="${CONTROLLER_ROOT}/controller.status"
SEEDS=(42 123 2026)

mkdir -p "${CONTROLLER_ROOT}"
cd "${PROJECT_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! command -v flock >/dev/null 2>&1; then
  printf 'nvidia-smi and flock are required\n' >&2
  exit 1
fi
exec 9> "${CONTROLLER_ROOT}/controller.lock"
if ! flock -n 9; then
  printf 'another three-GPU controller is already running\n' >&2
  exit 1
fi
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
fi
if [[ ! "${MAX_GPU_USED_MIB}" =~ ^[0-9]+$ ]]; then
  printf 'MAX_GPU_USED_MIB must be a non-negative integer\n' >&2
  exit 2
fi

timestamp() {
  date --iso-8601=seconds
}

pending_seeds() {
  local seed status_file result_file
  for seed in "${SEEDS[@]}"; do
    status_file="${OUTPUT_ROOT}/seed_${seed}/pipeline.status"
    result_file="${OUTPUT_ROOT}/seed_${seed}/evaluation/coco.json"
    if [[ ! -s "${result_file}" ]] ||
      [[ ! -f "${status_file}" ]] ||
      [[ "$(<"${status_file}")" != "complete" ]]; then
      printf '%s\n' "${seed}"
    fi
  done
}

available_gpus() {
  nvidia-smi \
    --query-gpu=index,memory.used \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F ',' -v maximum="${MAX_GPU_USED_MIB}" '
      {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        if (($2 + 0) <= maximum) print $1
      }
    '
}

attempt=0
while true; do
  mapfile -t pending < <(pending_seeds)
  if (( ${#pending[@]} == 0 )); then
    if ! "${PYTHON_BIN}" "${HERE}/summarize.py" \
      --input-root "${OUTPUT_ROOT}" \
      --output "${OUTPUT_ROOT}/summary.json"; then
      printf 'failed_summary checked_at=%s\n' "$(timestamp)" > "${CONTROLLER_STATUS}"
      exit 1
    fi
    printf 'complete completed_at=%s\n' "$(timestamp)" > "${CONTROLLER_STATUS}"
    exit 0
  fi

  mapfile -t gpus < <(available_gpus)
  if (( ${#gpus[@]} < ${#pending[@]} )); then
    printf 'waiting_for_gpus required=%s available=%s seeds=%s checked_at=%s\n' \
      "${#pending[@]}" "${#gpus[@]}" "${pending[*]}" "$(timestamp)" \
      > "${CONTROLLER_STATUS}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  attempt=$((attempt + 1))
  pids=()
  logs=()
  printf 'running attempt=%s seeds=%s gpus=%s started_at=%s\n' \
    "${attempt}" "${pending[*]}" "${gpus[*]:0:${#pending[@]}}" "$(timestamp)" \
    > "${CONTROLLER_STATUS}"

  for index in "${!pending[@]}"; do
    seed="${pending[$index]}"
    gpu="${gpus[$index]}"
    run_root="${OUTPUT_ROOT}/seed_${seed}"
    log_file="${run_root}/pipeline.log"
    mkdir -p "${run_root}"
    printf '\n[controller] attempt=%s seed=%s gpu=%s started_at=%s\n' \
      "${attempt}" "${seed}" "${gpu}" "$(timestamp)" >> "${log_file}"
    env \
      PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash "${HERE}/run_seed_pipeline.sh" "${seed}" "cuda:${gpu}" \
      >> "${log_file}" 2>&1 &
    pids+=("$!")
    logs+=("${log_file}")
    printf '%s\n' "$!" > "${run_root}/pipeline.pid"
  done

  retry_resource_failure=0
  fatal_failure=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      continue
    fi
    if tail -n 200 "${logs[$index]}" |
      grep -Eq 'OutOfMemoryError|CUDA out of memory|CUDA-capable device.*busy'; then
      retry_resource_failure=1
    else
      fatal_failure=1
    fi
  done

  if (( fatal_failure != 0 )); then
    printf 'failed_non_resource_error attempt=%s checked_at=%s\n' \
      "${attempt}" "$(timestamp)" > "${CONTROLLER_STATUS}"
    exit 1
  fi
  if (( retry_resource_failure != 0 )); then
    printf 'retrying_after_gpu_contention attempt=%s checked_at=%s\n' \
      "${attempt}" "$(timestamp)" > "${CONTROLLER_STATUS}"
    sleep "${POLL_SECONDS}"
  fi
done
