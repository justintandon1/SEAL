#!/usr/bin/env bash
# Evaluate 7B S_logic on MATH-500 / APPS / LogiQA (clean English-500).
#
# Protocol:
#   DeepSeek-R1-Distill-Qwen-7B, layer 20, coef -1.0, max_tokens 10000,
#   chat, --remove_bos, n=500 seed 42,
#   LogiQA data/LogiQA/eval_rand42_500_clean.json + --logiqa_english_only.
#
#   VECTOR=path/to/logiqa2_v_logic.pt DATASETS=logiqa bash scripts/eval_7b_s_logic.sh
#   VECTOR=... DATASETS=math,logiqa RUN_BASELINE=0 bash scripts/eval_7b_s_logic.sh 0
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

VECTOR=${VECTOR:?set VECTOR to the 7B S_logic .pt (dim 3584)}
export MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export LAYER="${LAYER:-20}"
export COEF="${COEF:--1.0}"
export MAX_TOKENS="${MAX_TOKENS:-10000}"
export SAMPLE_SEED="${SAMPLE_SEED:-42}"
export RESULT_ROOT="${RESULT_ROOT:-results/results_for_7b_logic_vectors}"
export DATASETS="${DATASETS:-logiqa,math,apps}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export LOGIQA_BATCH_SIZE="${LOGIQA_BATCH_SIZE:-8}"
export APPS_BATCH_SIZE="${APPS_BATCH_SIZE:-2}"
export RUN_BASELINE="${RUN_BASELINE:-1}"
export LOGIQA_SELECTION="${LOGIQA_SELECTION:-data/LogiQA/eval_rand42_500_clean.json}"
export LOGIQA_ENGLISH_ONLY="${LOGIQA_ENGLISH_ONLY:-1}"
export MATH_MAX_EXAMPLES="${MATH_MAX_EXAMPLES:-500}"
export APPS_MAX_EXAMPLES="${APPS_MAX_EXAMPLES:-500}"
export LOGIQA_MAX_EXAMPLES="${LOGIQA_MAX_EXAMPLES:-500}"
export APPS_SPLIT="${APPS_SPLIT:-test}"

exec bash scripts/eval_steering_vector_benchmarks.sh \
  "$VECTOR" \
  "${VECTOR_NAME:-logic_vector_7b}" \
  "${1:-0}"
