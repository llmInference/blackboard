#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-exp}"
WEBARENA_OUTPUT_ROOT="${WEBARENA_OUTPUT_ROOT:-$ROOT_DIR/experiment/webarena/outputs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WEBARENA_TASK_ID="${WEBARENA_TASK_ID:-}"
WEBARENA_LIMIT="${WEBARENA_LIMIT:-1}"
WEBARENA_DATASET_NAME="${WEBARENA_DATASET_NAME:-webarena-verified}"
TASK_SET_SMOKE="${WEBARENA_TASK_SET_SMOKE:-$ROOT_DIR/experiment/common/task_sets/webarena_script_browser_smoke.json}"
TASK_SET_COMPARE="${WEBARENA_TASK_SET_COMPARE:-$ROOT_DIR/experiment/common/task_sets/webarena_script_browser_smoke.json}"
TASK_SET_ABLATION="${WEBARENA_TASK_SET_ABLATION:-$ROOT_DIR/experiment/common/task_sets/webarena_script_browser_smoke.json}"
WEBARENA_CONFIG="${WEBARENA_CONFIG:-$ROOT_DIR/experiment/common/configs/webarena_verified_local.example.json}"
WEBARENA_MAX_STEPS="${WEBARENA_MAX_STEPS:-4}"
WEBARENA_MAX_KERNEL_STEPS="${WEBARENA_MAX_KERNEL_STEPS:-8}"
WEBARENA_MAX_NO_PROGRESS="${WEBARENA_MAX_NO_PROGRESS:-2}"
WEBARENA_HEADLESS="${WEBARENA_HEADLESS:-1}"
WEBARENA_SLOW_MO_MS="${WEBARENA_SLOW_MO_MS:-0}"
WEBARENA_COMPARE_SYSTEMS="${WEBARENA_COMPARE_SYSTEMS:-blackboard,blackboard_no_architect}"
WEBARENA_ABLATION_MODES="${WEBARENA_ABLATION_MODES:-full,no_architect}"
WEBARENA_LLM_BACKEND="${WEBARENA_LLM_BACKEND:-}"
WEBARENA_LLM_MODEL="${WEBARENA_LLM_MODEL:-}"
WEBARENA_LLM_MODEL_ENV="${WEBARENA_LLM_MODEL_ENV:-}"
WEBARENA_LLM_BASE_URL="${WEBARENA_LLM_BASE_URL:-}"
WEBARENA_LLM_BASE_URL_ENV="${WEBARENA_LLM_BASE_URL_ENV:-}"
WEBARENA_LLM_API_KEY_ENV="${WEBARENA_LLM_API_KEY_ENV:-}"
WEBARENA_LLM_TEMPERATURE="${WEBARENA_LLM_TEMPERATURE:-0.0}"
WEBARENA_LLM_TIMEOUT="${WEBARENA_LLM_TIMEOUT:-60.0}"

usage() {
  cat <<'EOF'
Usage:
  run_webarena_experiments.sh [target] [options]
  run_webarena_experiments.sh --target <target> [options]

Targets:
  smoke  Run the current WebArena Blackboard smoke slice
  compare  Run the current WebArena Blackboard system comparison slice
  ablation Run the current WebArena Blackboard ablation slice
  all      Run smoke, compare, then ablation

Options:
  --conda-env <name>            Conda env name (default: exp)
  --output-root <dir>           Output root directory
  --run-tag <tag>               Run tag suffix used in output dirs
  --config <file>               WebArena config json
  --task-set-smoke <file>       Task-set manifest for smoke
  --task-set-compare <file>     Task-set manifest for compare
  --task-set-ablation <file>    Task-set manifest for ablation
  --task-id <id>                Single task id override
  --limit <n>                   Max tasks to run for each target
  --dataset-name <name>         Dataset name recorded in outputs
  --max-steps <n>               Max neutral turns
  --max-kernel-steps <n>        Max kernel steps per turn
  --max-no-progress <n>         No-progress threshold
  --headless                    Run browser in headless mode
  --no-headless                 Run browser with UI
  --slow-mo-ms <n>              Playwright slow motion (ms)
  --compare-systems <csv>       Compare systems list
  --ablation-modes <csv>        Ablation modes list
  --llm-backend <name>          LLM backend
  --llm-model <name>            LLM model name
  --llm-model-env <ENV>         Env var name for model
  --llm-base-url <url>          LLM base URL
  --llm-base-url-env <ENV>      Env var name for base URL
  --llm-api-key-env <ENV>       Env var name for API key
  --llm-temperature <float>     LLM temperature
  --llm-timeout <seconds>       LLM timeout
  -h, --help                    Show this help

Examples:
  run_webarena_experiments.sh ablation --task-set-ablation /abs/webarena_non_map_50.json --limit 50
  run_webarena_experiments.sh --target all --config /abs/webarena_verified_local.json --no-headless
EOF
}

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

require_arg_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "[ERROR] ${flag} requires a value." >&2
    usage
    exit 1
  fi
}

parse_args() {
  TARGET="all"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      smoke|compare|ablation|all)
        TARGET="$1"
        shift
        ;;
      --target)
        require_arg_value "$1" "${2:-}"
        TARGET="$2"
        shift 2
        ;;
      --conda-env)
        require_arg_value "$1" "${2:-}"
        CONDA_ENV="$2"
        shift 2
        ;;
      --output-root)
        require_arg_value "$1" "${2:-}"
        WEBARENA_OUTPUT_ROOT="$2"
        shift 2
        ;;
      --run-tag)
        require_arg_value "$1" "${2:-}"
        RUN_TAG="$2"
        shift 2
        ;;
      --config)
        require_arg_value "$1" "${2:-}"
        WEBARENA_CONFIG="$2"
        shift 2
        ;;
      --task-set-smoke)
        require_arg_value "$1" "${2:-}"
        TASK_SET_SMOKE="$2"
        shift 2
        ;;
      --task-set-compare)
        require_arg_value "$1" "${2:-}"
        TASK_SET_COMPARE="$2"
        shift 2
        ;;
      --task-set-ablation)
        require_arg_value "$1" "${2:-}"
        TASK_SET_ABLATION="$2"
        shift 2
        ;;
      --task-id)
        require_arg_value "$1" "${2:-}"
        WEBARENA_TASK_ID="$2"
        shift 2
        ;;
      --limit)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LIMIT="$2"
        shift 2
        ;;
      --dataset-name)
        require_arg_value "$1" "${2:-}"
        WEBARENA_DATASET_NAME="$2"
        shift 2
        ;;
      --max-steps)
        require_arg_value "$1" "${2:-}"
        WEBARENA_MAX_STEPS="$2"
        shift 2
        ;;
      --max-kernel-steps)
        require_arg_value "$1" "${2:-}"
        WEBARENA_MAX_KERNEL_STEPS="$2"
        shift 2
        ;;
      --max-no-progress)
        require_arg_value "$1" "${2:-}"
        WEBARENA_MAX_NO_PROGRESS="$2"
        shift 2
        ;;
      --headless)
        WEBARENA_HEADLESS="1"
        shift
        ;;
      --no-headless)
        WEBARENA_HEADLESS="0"
        shift
        ;;
      --slow-mo-ms)
        require_arg_value "$1" "${2:-}"
        WEBARENA_SLOW_MO_MS="$2"
        shift 2
        ;;
      --compare-systems)
        require_arg_value "$1" "${2:-}"
        WEBARENA_COMPARE_SYSTEMS="$2"
        shift 2
        ;;
      --ablation-modes)
        require_arg_value "$1" "${2:-}"
        WEBARENA_ABLATION_MODES="$2"
        shift 2
        ;;
      --llm-backend)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_BACKEND="$2"
        shift 2
        ;;
      --llm-model)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_MODEL="$2"
        shift 2
        ;;
      --llm-model-env)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_MODEL_ENV="$2"
        shift 2
        ;;
      --llm-base-url)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_BASE_URL="$2"
        shift 2
        ;;
      --llm-base-url-env)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_BASE_URL_ENV="$2"
        shift 2
        ;;
      --llm-api-key-env)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_API_KEY_ENV="$2"
        shift 2
        ;;
      --llm-temperature)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_TEMPERATURE="$2"
        shift 2
        ;;
      --llm-timeout)
        require_arg_value "$1" "${2:-}"
        WEBARENA_LLM_TIMEOUT="$2"
        shift 2
        ;;
      -h|--help|help)
        usage
        exit 0
        ;;
      *)
        echo "[ERROR] Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

run_smoke() {
  local out_dir="${WEBARENA_OUTPUT_ROOT}/smoke_${RUN_TAG}"
  local headless_flag="--headless"
  if [[ "${WEBARENA_HEADLESS}" != "1" ]]; then
    headless_flag="--no-headless"
  fi
  log "[webarena-smoke] ${TASK_SET_SMOKE} -> ${out_dir}"
  env PYTHONPATH="${ROOT_DIR}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python \
      "${ROOT_DIR}/experiment/webarena/examples/run_blackboard_smoke.py" \
      --task-set-file "${TASK_SET_SMOKE}" \
      --config "${WEBARENA_CONFIG}" \
      --task-id "${WEBARENA_TASK_ID}" \
      --limit "${WEBARENA_LIMIT}" \
      --output-dir "${out_dir}" \
      --dataset-name "${WEBARENA_DATASET_NAME}" \
      --max-steps "${WEBARENA_MAX_STEPS}" \
      --max-kernel-steps "${WEBARENA_MAX_KERNEL_STEPS}" \
      --max-no-progress "${WEBARENA_MAX_NO_PROGRESS}" \
      --llm-backend "${WEBARENA_LLM_BACKEND}" \
      --llm-model "${WEBARENA_LLM_MODEL}" \
      --llm-model-env "${WEBARENA_LLM_MODEL_ENV}" \
      --llm-base-url "${WEBARENA_LLM_BASE_URL}" \
      --llm-base-url-env "${WEBARENA_LLM_BASE_URL_ENV}" \
      --llm-api-key-env "${WEBARENA_LLM_API_KEY_ENV}" \
      --llm-temperature "${WEBARENA_LLM_TEMPERATURE}" \
      --llm-timeout "${WEBARENA_LLM_TIMEOUT}" \
      --slow-mo-ms "${WEBARENA_SLOW_MO_MS}" \
      ${headless_flag} \
      --summary-only
}

run_compare() {
  local out_dir="${WEBARENA_OUTPUT_ROOT}/compare_${RUN_TAG}"
  local headless_flag="--headless"
  if [[ "${WEBARENA_HEADLESS}" != "1" ]]; then
    headless_flag="--no-headless"
  fi
  log "[webarena-compare] ${TASK_SET_COMPARE} -> ${out_dir}"
  env PYTHONPATH="${ROOT_DIR}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python \
      "${ROOT_DIR}/experiment/webarena/examples/run_system_compare.py" \
      --task-set-file "${TASK_SET_COMPARE}" \
      --config "${WEBARENA_CONFIG}" \
      --task-id "${WEBARENA_TASK_ID}" \
      --limit "${WEBARENA_LIMIT}" \
      --output-dir "${out_dir}" \
      --dataset-name "${WEBARENA_DATASET_NAME}" \
      --systems "${WEBARENA_COMPARE_SYSTEMS}" \
      --max-steps "${WEBARENA_MAX_STEPS}" \
      --max-kernel-steps "${WEBARENA_MAX_KERNEL_STEPS}" \
      --max-no-progress "${WEBARENA_MAX_NO_PROGRESS}" \
      --llm-backend "${WEBARENA_LLM_BACKEND}" \
      --llm-model "${WEBARENA_LLM_MODEL}" \
      --llm-model-env "${WEBARENA_LLM_MODEL_ENV}" \
      --llm-base-url "${WEBARENA_LLM_BASE_URL}" \
      --llm-base-url-env "${WEBARENA_LLM_BASE_URL_ENV}" \
      --llm-api-key-env "${WEBARENA_LLM_API_KEY_ENV}" \
      --llm-temperature "${WEBARENA_LLM_TEMPERATURE}" \
      --llm-timeout "${WEBARENA_LLM_TIMEOUT}" \
      --slow-mo-ms "${WEBARENA_SLOW_MO_MS}" \
      ${headless_flag}
}

run_ablation() {
  local out_dir="${WEBARENA_OUTPUT_ROOT}/ablation_${RUN_TAG}"
  local headless_flag="--headless"
  if [[ "${WEBARENA_HEADLESS}" != "1" ]]; then
    headless_flag="--no-headless"
  fi
  log "[webarena-ablation] ${TASK_SET_ABLATION} -> ${out_dir}"
  env PYTHONPATH="${ROOT_DIR}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python \
      "${ROOT_DIR}/experiment/webarena/examples/run_ablation.py" \
      --task-set-file "${TASK_SET_ABLATION}" \
      --config "${WEBARENA_CONFIG}" \
      --task-id "${WEBARENA_TASK_ID}" \
      --limit "${WEBARENA_LIMIT}" \
      --output-dir "${out_dir}" \
      --dataset-name "${WEBARENA_DATASET_NAME}" \
      --modes "${WEBARENA_ABLATION_MODES}" \
      --max-steps "${WEBARENA_MAX_STEPS}" \
      --max-kernel-steps "${WEBARENA_MAX_KERNEL_STEPS}" \
      --max-no-progress "${WEBARENA_MAX_NO_PROGRESS}" \
      --llm-backend "${WEBARENA_LLM_BACKEND}" \
      --llm-model "${WEBARENA_LLM_MODEL}" \
      --llm-model-env "${WEBARENA_LLM_MODEL_ENV}" \
      --llm-base-url "${WEBARENA_LLM_BASE_URL}" \
      --llm-base-url-env "${WEBARENA_LLM_BASE_URL_ENV}" \
      --llm-api-key-env "${WEBARENA_LLM_API_KEY_ENV}" \
      --llm-temperature "${WEBARENA_LLM_TEMPERATURE}" \
      --llm-timeout "${WEBARENA_LLM_TIMEOUT}" \
      --slow-mo-ms "${WEBARENA_SLOW_MO_MS}" \
      ${headless_flag}
}

parse_args "$@"
case "${TARGET}" in
  smoke)
    run_smoke
    ;;
  compare)
    run_compare
    ;;
  ablation)
    run_ablation
    ;;
  all)
    run_smoke
    run_compare
    run_ablation
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[ERROR] Unknown target: ${TARGET}" >&2
    usage
    exit 1
    ;;
esac
