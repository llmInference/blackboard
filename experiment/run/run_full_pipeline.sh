#!/usr/bin/env bash
# Full experiment pipeline: ALFWorld + ScienceWorld + WebArena
#
# Usage:
#   export OPENAI_API_BASE='https://tb.api.mkeai.com/v1'
#   export OPENAI_API_KEY='<your-key>'
#   export MODEL_NAME='deepseek-v3'
#   export ALFWORLD_DATA="$HOME/.cache/alfworld"
#   bash run_full_pipeline.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_NAME="${MODEL_NAME:-deepseek-v3}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://tb.api.mkeai.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_TIMEOUT="${OPENAI_API_TIMEOUT:-300}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export CONDA_ENV="${CONDA_ENV:-blackboard}"

LOG_DIR="${ROOT_DIR}/experiment/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_${RUN_TAG}.log"

# Tee all output to log file
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

[[ -n "${OPENAI_API_KEY}" ]] || { echo "[ERROR] OPENAI_API_KEY is required." >&2; exit 1; }

log "[pipeline] log -> ${LOG_FILE}"
log "[pipeline] model=${MODEL_NAME} base=${OPENAI_API_BASE} data=${ALFWORLD_DATA}"

log "[alfworld] starting"
bash "${ROOT_DIR}/experiment/run/run_alfworld_experiments.sh" all
log "[alfworld] done"

log "[scienceworld] starting"
bash "${ROOT_DIR}/experiment/run/run_scienceworld_experiments.sh" all
log "[scienceworld] done"

log "[webarena] starting"
bash "${ROOT_DIR}/experiment/run/run_webarena_experiments.sh" all
log "[webarena] done"

log "Pipeline complete."
