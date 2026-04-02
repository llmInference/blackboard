#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-exp}"

source "${ROOT_DIR}/experiment/run/export_experiment_env.sh"

WEBARENA_OUTPUT_ROOT="${WEBARENA_OUTPUT_ROOT:-${ROOT_DIR}/experiment/webarena/outputs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WEBARENA_TASK_SET_SMOKE="${WEBARENA_TASK_SET_SMOKE:-${ROOT_DIR}/experiment/common/task_sets/webarena_script_browser_smoke.json}"
WEBARENA_CONFIG="${WEBARENA_CONFIG:-${ROOT_DIR}/experiment/common/configs/webarena_verified_local.example.json}"
WEBARENA_MAX_STEPS="${WEBARENA_MAX_STEPS:-4}"
WEBARENA_MAX_KERNEL_STEPS="${WEBARENA_MAX_KERNEL_STEPS:-8}"
WEBARENA_MAX_NO_PROGRESS="${WEBARENA_MAX_NO_PROGRESS:-2}"
WEBARENA_HEADLESS="${WEBARENA_HEADLESS:-1}"

WEBARENA_LLM_BACKEND="${WEBARENA_LLM_BACKEND:-openai_compatible}"
WEBARENA_LLM_MODEL="${WEBARENA_LLM_MODEL:-${MODEL_NAME:-}}"
WEBARENA_LLM_BASE_URL="${WEBARENA_LLM_BASE_URL:-${OPENAI_API_BASE:-}}"
WEBARENA_LLM_API_KEY="${WEBARENA_LLM_API_KEY:-${OPENAI_API_KEY:-}}"
WEBARENA_LLM_TEMPERATURE="${WEBARENA_LLM_TEMPERATURE:-0.0}"
WEBARENA_LLM_TIMEOUT="${WEBARENA_LLM_TIMEOUT:-60.0}"

usage() {
  cat <<'EOF'
Usage:
  experiment/run/run_webarena_llm_smoke.sh

Environment overrides:
  CONDA_ENV=exp
  RUN_TAG=custom_tag
  WEBARENA_TASK_SET_SMOKE=/abs/path/to/manifest.json
  WEBARENA_CONFIG=/abs/path/to/webarena_config.json
  WEBARENA_OUTPUT_ROOT=/abs/path/to/output_root
  WEBARENA_MAX_STEPS=4
  WEBARENA_MAX_KERNEL_STEPS=8
  WEBARENA_MAX_NO_PROGRESS=2
  WEBARENA_HEADLESS=1
  WEBARENA_LLM_BACKEND=openai_compatible
  WEBARENA_LLM_MODEL=openai/gpt-5.3-codex
  WEBARENA_LLM_BASE_URL=https://api.qnaigc.com/v1
  WEBARENA_LLM_API_KEY=...

Notes:
  - 默认复用 experiment/run/export_experiment_env.sh 里的 MODEL_NAME / OPENAI_API_BASE / OPENAI_API_KEY
  - 如果显式设置 WEBARENA_LLM_MODEL / WEBARENA_LLM_BASE_URL / WEBARENA_LLM_API_KEY，会覆盖默认值
  - 该脚本只跑 WebArena LLM smoke，不跑 compare / ablation
EOF
}

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${WEBARENA_LLM_MODEL}" || -z "${WEBARENA_LLM_BASE_URL}" || -z "${WEBARENA_LLM_API_KEY}" ]]; then
  echo "[webarena-llm-smoke] missing llm model/base-url/api-key configuration" >&2
  exit 1
fi

OUT_DIR="${WEBARENA_OUTPUT_ROOT}/llm_smoke_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

HEADLESS_FLAG="--headless"
if [[ "${WEBARENA_HEADLESS}" != "1" ]]; then
  HEADLESS_FLAG="--no-headless"
fi

log "[webarena-llm-smoke] out_dir=${OUT_DIR}"
log "[webarena-llm-smoke] task_set=${WEBARENA_TASK_SET_SMOKE}"
log "[webarena-llm-smoke] config=${WEBARENA_CONFIG}"
log "[webarena-llm-smoke] model=${WEBARENA_LLM_MODEL}"

env \
  PYTHONPATH="${ROOT_DIR}" \
  WEBARENA_LLM_MODEL="${WEBARENA_LLM_MODEL}" \
  WEBARENA_LLM_BASE_URL="${WEBARENA_LLM_BASE_URL}" \
  WEBARENA_LLM_API_KEY="${WEBARENA_LLM_API_KEY}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python \
    "${ROOT_DIR}/experiment/webarena/examples/run_blackboard_smoke.py" \
    --task-set-file "${WEBARENA_TASK_SET_SMOKE}" \
    --config "${WEBARENA_CONFIG}" \
    --output-dir "${OUT_DIR}" \
    --max-steps "${WEBARENA_MAX_STEPS}" \
    --max-kernel-steps "${WEBARENA_MAX_KERNEL_STEPS}" \
    --max-no-progress "${WEBARENA_MAX_NO_PROGRESS}" \
    --llm-backend "${WEBARENA_LLM_BACKEND}" \
    --llm-model-env WEBARENA_LLM_MODEL \
    --llm-base-url-env WEBARENA_LLM_BASE_URL \
    --llm-api-key-env WEBARENA_LLM_API_KEY \
    --llm-temperature "${WEBARENA_LLM_TEMPERATURE}" \
    --llm-timeout "${WEBARENA_LLM_TIMEOUT}" \
    ${HEADLESS_FLAG} \
    --summary-only

log "[webarena-llm-smoke] summary=${OUT_DIR}/summary.json"
