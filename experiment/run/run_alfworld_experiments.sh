#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-blackboard}"
MODEL_NAME="${MODEL_NAME:-deepseek-v3}"
OPENAI_API_BASE="${OPENAI_API_BASE:-https://tb.api.mkeai.com/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-300}"
ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
OUTPUT_ROOT="${ALFWORLD_OUTPUT_ROOT:-$ROOT_DIR/experiment/alfworld/outputs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

TASK_SET_DEBUG="${ALFWORLD_TASK_SET_DEBUG:-$ROOT_DIR/experiment/common/task_sets/alfworld_debug.json}"
TASK_SET_FORMAL="${ALFWORLD_TASK_SET_FORMAL:-$ROOT_DIR/experiment/common/task_sets/alfworld_formal.json}"

REQUIRED_PYTHONPATH="$ROOT_DIR:$ROOT_DIR/alfworld:$ROOT_DIR/blackboard/libs/kernel_system:$ROOT_DIR/blackboard/libs/blackboard:$ROOT_DIR/langgraph/libs/langgraph:$ROOT_DIR/experiment/alfworld"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REQUIRED_PYTHONPATH}:${PYTHONPATH}"
else
  export PYTHONPATH="${REQUIRED_PYTHONPATH}"
fi

usage() {
  cat <<'EOF'
Usage:
  run_alfworld_experiments.sh [target]

Targets:
  exp1  Experiment 1: blackboard/langgraph/autogen compare + judge
  exp2  Experiment 2: state capture + schema guard evaluation
  exp3  Experiment 3: blackboard/langgraph/autogen/ablate_c4 compare
  exp4  Experiment 4: blackboard/langgraph/autogen formal compare
  exp5  Experiment 5: blackboard/langgraph/autogen debug compare
  exp6  Experiment 6: blackboard full vs ablate_c1~c5
  all   Run all experiments sharing one debug task set run

Defaults:
  MODEL_NAME=deepseek-v3
  OPENAI_API_BASE=https://tb.api.mkeai.com/v1
  OPENAI_API_KEY=
  ALFWORLD_DATA=$HOME/.cache/alfworld
EOF
}

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

RUN() {
  env PYTHONPATH="${PYTHONPATH}" \
    OPENAI_API_BASE="${OPENAI_API_BASE}" \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT}" \
    ALFWORLD_DATA="${ALFWORLD_DATA}" \
    ALFWORLD_LLM_MODEL="${MODEL_NAME}" \
    ALFWORLD_LLM_BASE_URL="${OPENAI_API_BASE}" \
    ALFWORLD_LLM_API_KEY="${OPENAI_API_KEY}" \
    conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
}

run_comm_judge() {
  local judge_input="$1"
  [[ -f "${judge_input}" ]] || return 0
  RUN "$ROOT_DIR/experiment/common/run_communication_judge.py" \
    --input-path "${judge_input}" \
    --backend openai_compatible \
    --model "${MODEL_NAME}" \
    --base-url "${OPENAI_API_BASE}" \
    --api-key "${OPENAI_API_KEY}" \
    --timeout "${OPENAI_API_TIMEOUT}" \
    --sleep-between-requests 4.5 \
    --continue-on-error
}

common_llm_args() {
  printf '%s\n' \
    --llm-backend openai_compatible \
    --llm-model-env ALFWORLD_LLM_MODEL \
    --llm-base-url-env ALFWORLD_LLM_BASE_URL \
    --llm-api-key-env ALFWORLD_LLM_API_KEY \
    --llm-timeout "${OPENAI_API_TIMEOUT}"
}

run_exp1() {
  local task_set="${ALFWORLD_EXP1_TASK_SET:-$TASK_SET_DEBUG}"
  local out_dir="${OUTPUT_ROOT}/exp1_alfworld_comm_${RUN_TAG}"
  local max_steps="${ALFWORLD_EXP1_MAX_STEPS:-20}"

  log "[exp1] system compare on ${task_set} -> ${out_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_system_compare.py" \
    --task-set-file "${task_set}" \
    --output-dir "${out_dir}" \
    --systems blackboard,langgraph,autogen \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    $(common_llm_args)

  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${out_dir}/system_compare_summary.json"

  run_comm_judge "${out_dir}/comm_judge.jsonl"
}

run_exp2() {
  local task_set="${ALFWORLD_EXP2_TASK_SET:-$TASK_SET_DEBUG}"
  local capture_dir="${OUTPUT_ROOT}/exp2_alfworld_capture_${RUN_TAG}"
  local schema_dir="${OUTPUT_ROOT}/exp2_alfworld_schema_guard_${RUN_TAG}"
  local run_id="capture_states_${RUN_TAG}"
  local max_steps="${ALFWORLD_EXP2_MAX_STEPS:-20}"

  log "[exp2] capture states on ${task_set} -> ${capture_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_capture_states.py" \
    --task-set-file "${task_set}" \
    --output-dir "${capture_dir}" \
    --run-id "${run_id}" \
    --max-steps "${max_steps}" \
    --model-name "${MODEL_NAME}" \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    $(common_llm_args)

  log "[exp2] schema guard -> ${schema_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_schema_guard_eval.py" \
    --states-file "${capture_dir}/captured_states.jsonl" \
    --output-dir "${schema_dir}"
}

run_exp3() {
  local task_set="${ALFWORLD_EXP3_TASK_SET:-$TASK_SET_DEBUG}"
  local out_dir="${OUTPUT_ROOT}/exp3_alfworld_context_${RUN_TAG}"
  local max_steps="${ALFWORLD_EXP3_MAX_STEPS:-20}"

  log "[exp3] system compare on ${task_set} -> ${out_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_system_compare.py" \
    --task-set-file "${task_set}" \
    --output-dir "${out_dir}" \
    --systems blackboard,langgraph,autogen \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    $(common_llm_args)

  RUN "$ROOT_DIR/experiment/alfworld/examples/run_system_compare.py" \
    --task-set-file "${task_set}" \
    --output-dir "${out_dir}" \
    --systems blackboard \
    --blackboard-mode ablate_c4 \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    $(common_llm_args)

  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${out_dir}/system_compare_summary.json"
}

run_exp4() {
  local task_set="${ALFWORLD_EXP4_TASK_SET:-$TASK_SET_FORMAL}"
  local out_dir="${OUTPUT_ROOT}/exp4_alfworld_generalization_${RUN_TAG}"
  local max_steps="${ALFWORLD_EXP4_MAX_STEPS:-20}"

  log "[exp4] system compare on ${task_set} -> ${out_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_system_compare.py" \
    --task-set-file "${task_set}" \
    --output-dir "${out_dir}" \
    --systems blackboard,langgraph,autogen \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    --resume \
    $(common_llm_args)

  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${out_dir}/system_compare_summary.json"
}

run_exp5() {
  local task_set="${ALFWORLD_EXP5_TASK_SET:-$TASK_SET_DEBUG}"
  local out_dir="${OUTPUT_ROOT}/exp5_alfworld_accuracy_${RUN_TAG}"
  local max_steps="${ALFWORLD_EXP5_MAX_STEPS:-20}"

  log "[exp5] system compare on ${task_set} -> ${out_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_system_compare.py" \
    --task-set-file "${task_set}" \
    --output-dir "${out_dir}" \
    --systems blackboard,langgraph,autogen \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    --resume \
    $(common_llm_args)

  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${out_dir}/system_compare_summary.json"
}

run_exp6() {
  local task_set="${ALFWORLD_EXP6_TASK_SET:-$TASK_SET_DEBUG}"
  local out_dir="${OUTPUT_ROOT}/exp6_alfworld_ablation_${RUN_TAG}"
  local max_steps="${ALFWORLD_EXP6_MAX_STEPS:-20}"

  log "[exp6] component ablation on ${task_set} -> ${out_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_ablation.py" \
    --task-set-file "${task_set}" \
    --output-dir "${out_dir}" \
    --modes full,ablate_c1,ablate_c2,ablate_c3,ablate_c4,ablate_c5,ablate_c6 \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    $(common_llm_args)

  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${out_dir}/ablation_summary.json"
}

# run_all: each system/mode is run only once on the debug task set.
# shared_dir holds ablation (all blackboard modes) + system_compare (langgraph, autogen).
# exp1/exp3/exp5 reuse shared results; exp2 runs capture_states separately; exp4 uses formal task set.
run_all() {
  local task_set="${TASK_SET_DEBUG}"
  local max_steps="${ALFWORLD_ALL_MAX_STEPS:-20}"
  local shared_dir="${OUTPUT_ROOT}/shared_debug_${RUN_TAG}"

  # Step 1: all blackboard modes (covers exp1/exp3/exp5/exp6)
  log "[all] ablation full+ablate_c1~c6 -> ${shared_dir}/ablation"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_ablation.py" \
    --task-set-file "${task_set}" \
    --output-dir "${shared_dir}/ablation" \
    --modes full,ablate_c1,ablate_c2,ablate_c3,ablate_c4,ablate_c5,ablate_c6 \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    $(common_llm_args)

  # Step 2: langgraph + autogen, reuse same gamefiles (covers exp1/exp3/exp5)
  log "[all] system compare langgraph+autogen -> ${shared_dir}/system_compare"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_system_compare.py" \
    --gamefiles-file "${shared_dir}/ablation/selected_gamefiles.json" \
    --output-dir "${shared_dir}/system_compare" \
    --systems langgraph,autogen \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    $(common_llm_args)

  # Step 3: exp2 capture states (needs states file, cannot reuse ablation)
  local capture_dir="${shared_dir}/exp2_capture"
  local run_id="capture_states_${RUN_TAG}"
  log "[all/exp2] capture states -> ${capture_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_capture_states.py" \
    --gamefiles-file "${shared_dir}/ablation/selected_gamefiles.json" \
    --output-dir "${capture_dir}" \
    --run-id "${run_id}" \
    --max-steps "${max_steps}" \
    --model-name "${MODEL_NAME}" \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    $(common_llm_args)

  log "[all/exp2] schema guard"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_schema_guard_eval.py" \
    --states-file "${capture_dir}/captured_states.jsonl" \
    --output-dir "${shared_dir}/exp2_schema_guard"

  # Step 4: exp4 formal task set (different gamefiles, must run separately)
  local formal_dir="${OUTPUT_ROOT}/exp4_alfworld_generalization_${RUN_TAG}"
  log "[all/exp4] formal system compare -> ${formal_dir}"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_system_compare.py" \
    --task-set-file "${TASK_SET_FORMAL}" \
    --output-dir "${formal_dir}" \
    --systems blackboard,langgraph,autogen \
    --workflow-mode planner_llm_action \
    --architect-mode llm \
    --model-name "${MODEL_NAME}" \
    --max-steps "${max_steps}" \
    --resume \
    $(common_llm_args)

  # Analysis
  log "[all] analysis"
  # exp1: system_compare (blackboard full is in ablation, langgraph/autogen in system_compare)
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${shared_dir}/ablation/ablation_summary.json"
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${shared_dir}/system_compare/system_compare_summary.json"

  # exp6
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${shared_dir}/ablation/ablation_summary.json" \
    --output-file "${shared_dir}/ablation/ablation_summary_exp6_analysis.json"

  # exp4
  RUN "$ROOT_DIR/experiment/alfworld/examples/run_result_analysis.py" \
    --summary-file "${formal_dir}/system_compare_summary.json"

  run_comm_judge "${shared_dir}/system_compare/comm_judge.jsonl"

  log "[all] done. shared results in ${shared_dir}"
}

log "[config] model=${MODEL_NAME} base=${OPENAI_API_BASE} data=${ALFWORLD_DATA}"

target="${1:-all}"
case "${target}" in
  exp1) run_exp1 ;;
  exp2) run_exp2 ;;
  exp3) run_exp3 ;;
  exp4) run_exp4 ;;
  exp5) run_exp5 ;;
  exp6) run_exp6 ;;
  all) run_all ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown target: ${target}" >&2
    usage
    exit 1
    ;;
esac
