#!/usr/bin/env bash

export OPENAI_API_BASE="https://api.qnaigc.com/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export MODEL_NAME="openai/gpt-5.3-codex"

export ALFWORLD_DATA="$HOME/.cache/alfworld"
