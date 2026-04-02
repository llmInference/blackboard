#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-exp}"
OUTPUT_ROOT="${TAU2BENCH_OUTPUT_ROOT:-$ROOT_DIR/outputs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

TAU2BENCH_PROFILE="${TAU2BENCH_PROFILE:-formal}" #formal debug
TAU2BENCH_TASK_SET_DEBUG="${TAU2BENCH_TASK_SET_DEBUG:-$ROOT_DIR/experiment/common/task_sets/tau2bench_debug.json}"
TAU2BENCH_TASK_SET_FORMAL="${TAU2BENCH_TASK_SET_FORMAL:-$ROOT_DIR/experiment/common/task_sets/tau2bench_formal.json}"
TAU2BENCH_TASK_SET_FILE="${TAU2BENCH_TASK_SET_FILE:-}"
TAU2BENCH_ABLATION_MODES="${TAU2BENCH_ABLATION_MODES:-full,ablate_c1,ablate_c2,ablate_c3,ablate_c4}"
TAU2BENCH_LIMIT="${TAU2BENCH_LIMIT:-20}"
TAU2BENCH_LIMIT_DEBUG="${TAU2BENCH_LIMIT_DEBUG:-1}"
TAU2BENCH_LIMIT_FORMAL="${TAU2BENCH_LIMIT_FORMAL:-10}"
TAU2BENCH_SEED="${TAU2BENCH_SEED:-42}"
TAU2BENCH_MAX_STEPS="${TAU2BENCH_MAX_STEPS:-50}"
TAU2BENCH_MAX_KERNEL_STEPS="${TAU2BENCH_MAX_KERNEL_STEPS:-50}"
TAU2BENCH_USER_IMPL="${TAU2BENCH_USER_IMPL:-user_simulator}"
TAU2BENCH_USER_LLM="${TAU2BENCH_USER_LLM:-openai/deepseek-v3}"

if [[ -z "${TAU2BENCH_TASK_SET_FILE}" ]]; then
  case "${TAU2BENCH_PROFILE}" in
    debug)
      TAU2BENCH_TASK_SET_FILE="${TAU2BENCH_TASK_SET_DEBUG}"
      ;;
    formal)
      TAU2BENCH_TASK_SET_FILE="${TAU2BENCH_TASK_SET_FORMAL}"
      ;;
    *)
      echo "[tau2bench] unknown TAU2BENCH_PROFILE=${TAU2BENCH_PROFILE}; expected debug or formal" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${TAU2BENCH_LIMIT:-}" ]]; then
  if [[ "${TAU2BENCH_PROFILE}" == "formal" ]]; then
    TAU2BENCH_LIMIT="${TAU2BENCH_LIMIT_FORMAL}"
  else
    TAU2BENCH_LIMIT="${TAU2BENCH_LIMIT_DEBUG}"
  fi
fi

source "$ROOT_DIR/experiment/run/export_experiment_env.sh"

REQUIRED_PYTHONPATH="$ROOT_DIR:$ROOT_DIR/blackboard/libs/kernel_system:$ROOT_DIR/langgraph/libs/langgraph"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REQUIRED_PYTHONPATH}:${PYTHONPATH}"
else
  export PYTHONPATH="${REQUIRED_PYTHONPATH}"
fi

OUT_DIR="${OUTPUT_ROOT}/tau2bench_blackboard_ablation_${RUN_TAG}"

echo "[tau2bench] profile=${TAU2BENCH_PROFILE}"
echo "[tau2bench] task_set=${TAU2BENCH_TASK_SET_FILE}"
echo "[tau2bench] output_dir=${OUT_DIR}"
echo "[tau2bench] modes=${TAU2BENCH_ABLATION_MODES}"
echo "[tau2bench] worker_model=${MODEL_NAME}"
echo "[tau2bench] user_model=${TAU2BENCH_USER_LLM}"
echo "[tau2bench] limit=${TAU2BENCH_LIMIT}"

conda run --no-capture-output -n "${CONDA_ENV}" python \
  "$ROOT_DIR/experiment/tau2bench/examples/run_ablation.py" \
  --task-set-file "${TAU2BENCH_TASK_SET_FILE}" \
  --output-dir "${OUT_DIR}" \
  --modes "${TAU2BENCH_ABLATION_MODES}" \
  --limit "${TAU2BENCH_LIMIT}" \
  --seed "${TAU2BENCH_SEED}" \
  --max-steps "${TAU2BENCH_MAX_STEPS}" \
  --max-kernel-steps "${TAU2BENCH_MAX_KERNEL_STEPS}" \
  --user "${TAU2BENCH_USER_IMPL}" \
  --llm-model "${MODEL_NAME}" \
  --llm-base-url "${OPENAI_API_BASE}" \
  --llm-temperature 1 \
  --user-llm "${TAU2BENCH_USER_LLM}"

echo "[tau2bench] ablation_summary=${OUT_DIR}/ablation_summary.json"
