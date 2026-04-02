#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-exp}"
OUTPUT_ROOT="${APPWORLD_OUTPUT_ROOT:-$ROOT_DIR/experiment/appworld/outputs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

APPWORLD_ROOT="${APPWORLD_ROOT:-$ROOT_DIR/appworld}"
APPWORLD_REMOTE_APIS_URL="${APPWORLD_REMOTE_APIS_URL:-http://127.0.0.1:9105}"
APPWORLD_DATASET="${APPWORLD_DATASET:-dev}"
APPWORLD_LIMIT="${APPWORLD_LIMIT:-3}"
APPWORLD_TASK_IDS="${APPWORLD_TASK_IDS:-}"
APPWORLD_TASK_FILE="${APPWORLD_TASK_FILE:-}"
APPWORLD_EXPERIMENT_NAME="${APPWORLD_EXPERIMENT_NAME:-appworld_blackboard_small_scale}"
APPWORLD_MAX_STEPS="${APPWORLD_MAX_STEPS:-20}"
APPWORLD_MAX_KERNEL_STEPS="${APPWORLD_MAX_KERNEL_STEPS:-20}"
APPWORLD_MAX_NO_PROGRESS="${APPWORLD_MAX_NO_PROGRESS:-1}"
APPWORLD_WALL_CLOCK_TIMEOUT="${APPWORLD_WALL_CLOCK_TIMEOUT:-60}"
APPWORLD_TIMEOUT="${APPWORLD_TIMEOUT:-60}"
APPWORLD_TEMPERATURE="${APPWORLD_TEMPERATURE:-0}"
APPWORLD_MAX_TOKENS="${APPWORLD_MAX_TOKENS:-1200}"
APPWORLD_RESUME="${APPWORLD_RESUME:-1}"
APPWORLD_LOAD_GROUND_TRUTH="${APPWORLD_LOAD_GROUND_TRUTH:-1}"
APPWORLD_GROUND_TRUTH_MODE="${APPWORLD_GROUND_TRUTH_MODE:-minimal}"

source "$ROOT_DIR/experiment/run/export_experiment_env.sh"

MODEL_NAME="${APPWORLD_LLM_MODEL:-${MODEL_NAME:-}}"
OPENAI_API_BASE="${APPWORLD_LLM_BASE_URL:-${OPENAI_API_BASE:-}}"
OPENAI_API_KEY="${APPWORLD_LLM_API_KEY:-${OPENAI_API_KEY:-}}"

OUT_DIR="${OUTPUT_ROOT}/appworld_small_scale_${RUN_TAG}"
SUMMARY_JSONL="${OUT_DIR}/task_summaries.jsonl"
STATUS_TSV="${OUT_DIR}/task_status.tsv"
TASKS_TXT="${OUT_DIR}/tasks.txt"

usage() {
  cat <<'EOF'
Usage:
  experiment/run/run_appworld_small_scale.sh

Environment overrides:
  CONDA_ENV=exp
  APPWORLD_ROOT=/path/to/appworld
  APPWORLD_REMOTE_APIS_URL=http://127.0.0.1:9105
  APPWORLD_DATASET=dev
  APPWORLD_LIMIT=3
  APPWORLD_TASK_IDS=50e1ac9_1,xxxx
  APPWORLD_TASK_FILE=/path/to/task_ids.txt
  APPWORLD_EXPERIMENT_NAME=appworld_blackboard_small_scale
  APPWORLD_MAX_STEPS=20
  APPWORLD_MAX_KERNEL_STEPS=20
  APPWORLD_MAX_NO_PROGRESS=1
  APPWORLD_WALL_CLOCK_TIMEOUT=60
  APPWORLD_TIMEOUT=60
  APPWORLD_LLM_MODEL=openai/gpt-5.3-codex
  APPWORLD_LLM_BASE_URL=https://api.qnaigc.com/v1
  APPWORLD_LLM_API_KEY=...
  APPWORLD_RESUME=1
  APPWORLD_LOAD_GROUND_TRUTH=1
  APPWORLD_GROUND_TRUTH_MODE=minimal

Notes:
  - If APPWORLD_TASK_IDS is set, it takes precedence.
  - Else if APPWORLD_TASK_FILE is set, non-empty non-comment lines are used.
  - Else task ids are loaded from the selected AppWorld dataset and truncated by APPWORLD_LIMIT.
EOF
}

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${MODEL_NAME}" || -z "${OPENAI_API_BASE}" || -z "${OPENAI_API_KEY}" ]]; then
  echo "[appworld] missing model/base-url/api-key configuration" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
: > "${SUMMARY_JSONL}"
printf 'task_id\tstatus\tfinished\tused_fallback_architect\tfinish_reason\tkernel_status\teval_success\ttask_goal_completion\tevaluation_error\toutput_json\tstderr_log\n' > "${STATUS_TSV}"

collect_task_ids() {
  if [[ -n "${APPWORLD_TASK_IDS}" ]]; then
    printf '%s\n' "${APPWORLD_TASK_IDS}" | tr ',' '\n' | sed '/^[[:space:]]*$/d'
    return 0
  fi

  if [[ -n "${APPWORLD_TASK_FILE}" ]]; then
    sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "${APPWORLD_TASK_FILE}"
    return 0
  fi

  APPWORLD_ROOT_VALUE="${APPWORLD_ROOT}" DATASET_NAME="${APPWORLD_DATASET}" LIMIT_VALUE="${APPWORLD_LIMIT}" conda run --no-capture-output -n "${CONDA_ENV}" python -c '
from appworld import load_task_ids, update_root
import os
update_root(os.environ["APPWORLD_ROOT_VALUE"])
dataset = os.environ["DATASET_NAME"]
limit = int(os.environ["LIMIT_VALUE"])
for task_id in load_task_ids(dataset)[:limit]:
    print(task_id)
'
}

mapfile -t TASK_IDS < <(collect_task_ids)

if [[ "${#TASK_IDS[@]}" -eq 0 ]]; then
  echo "[appworld] no task ids resolved" >&2
  exit 1
fi

if [[ -z "${APPWORLD_TASK_IDS}" && -z "${APPWORLD_TASK_FILE}" ]]; then
  TASK_IDS=("${TASK_IDS[@]:0:${APPWORLD_LIMIT}}")
fi

printf '%s\n' "${TASK_IDS[@]}" > "${TASKS_TXT}"

log "[appworld] out_dir=${OUT_DIR}"
log "[appworld] dataset=${APPWORLD_DATASET}"
log "[appworld] tasks=${#TASK_IDS[@]}"
log "[appworld] model=${MODEL_NAME}"
log "[appworld] remote_apis_url=${APPWORLD_REMOTE_APIS_URL}"
if [[ "${APPWORLD_LOAD_GROUND_TRUTH}" == "1" ]]; then
  GROUND_TRUTH_FLAG="--load-ground-truth"
else
  GROUND_TRUTH_FLAG="--no-load-ground-truth"
fi

for task_id in "${TASK_IDS[@]}"; do
  out_json="${OUT_DIR}/${task_id}.json"
  err_log="${OUT_DIR}/${task_id}.stderr.log"
  tmp_json="${out_json}.tmp"
  tmp_err="${err_log}.tmp"

  if [[ "${APPWORLD_RESUME}" == "1" && -s "${out_json}" ]]; then
    log "[appworld] skip existing ${task_id}"
  else
    log "[appworld] running ${task_id}"
    rm -f "${tmp_json}" "${tmp_err}"
    if conda run --no-capture-output -n "${CONDA_ENV}" python \
      "${ROOT_DIR}/experiment/appworld/examples/run_blackboard_smoke.py" \
      --task-id "${task_id}" \
      --dataset-name "${APPWORLD_DATASET}" \
      --experiment-name "${APPWORLD_EXPERIMENT_NAME}" \
      --remote-apis-url "${APPWORLD_REMOTE_APIS_URL}" \
      --appworld-root "${APPWORLD_ROOT}" \
      --llm-model "${MODEL_NAME}" \
      --llm-base-url "${OPENAI_API_BASE}" \
      --llm-api-key "${OPENAI_API_KEY}" \
      --timeout "${APPWORLD_TIMEOUT}" \
      --temperature "${APPWORLD_TEMPERATURE}" \
      --max-tokens "${APPWORLD_MAX_TOKENS}" \
      --max-steps "${APPWORLD_MAX_STEPS}" \
      --max-kernel-steps "${APPWORLD_MAX_KERNEL_STEPS}" \
      --max-no-progress "${APPWORLD_MAX_NO_PROGRESS}" \
      --wall-clock-timeout "${APPWORLD_WALL_CLOCK_TIMEOUT}" \
      "${GROUND_TRUTH_FLAG}" \
      --ground-truth-mode "${APPWORLD_GROUND_TRUTH_MODE}" \
      --summary-only > "${tmp_json}" 2> "${tmp_err}"; then
      mv "${tmp_json}" "${out_json}"
      mv "${tmp_err}" "${err_log}"
    else
      status_code=$?
      mv "${tmp_json}" "${out_json}" 2>/dev/null || true
      mv "${tmp_err}" "${err_log}" 2>/dev/null || true
      printf '%s\tfailed\tfalse\t\t\t\t\t\t\t%s\t%s\n' "${task_id}" "${out_json}" "${err_log}" >> "${STATUS_TSV}"
      log "[appworld] task ${task_id} failed with exit code ${status_code}; see ${err_log}"
      continue
    fi
  fi

  OUT_JSON="${out_json}" SUMMARY_JSONL="${SUMMARY_JSONL}" STATUS_TSV="${STATUS_TSV}" python -c '
import json
import os
from pathlib import Path

out_json = Path(os.environ["OUT_JSON"])
summary_jsonl = Path(os.environ["SUMMARY_JSONL"])
status_tsv = Path(os.environ["STATUS_TSV"])

with out_json.open("r", encoding="utf-8") as f:
    payload = json.load(f)

evaluation = dict(payload.get("evaluation") or {})
evaluation_error = str(evaluation.get("evaluation_error", "") or "").replace("\t", " ").replace("\n", " ")

with summary_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

row = [
    str(payload.get("task_id", "")),
    "ok",
    str(payload.get("finished", "")),
    str(payload.get("used_fallback_architect", "")),
    str(payload.get("finish_reason", "")),
    str(payload.get("kernel_status", "")),
    str(evaluation.get("success", "")),
    str(evaluation.get("task_goal_completion", "")),
    evaluation_error,
    str(out_json),
    str(out_json.with_suffix(".stderr.log")),
]
with status_tsv.open("a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
'
done

log "[appworld] tasks_file=${TASKS_TXT}"
log "[appworld] status_tsv=${STATUS_TSV}"
log "[appworld] summaries_jsonl=${SUMMARY_JSONL}"
