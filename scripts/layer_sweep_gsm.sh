#!/usr/bin/env bash
# Layer-sweep study for the MATH steering vector (Justin's 7/18 task):
#
#   Phase 1  MATH-train baseline traces            (reused if present)
#   Phase 2  All-layer hidden-state extraction     (the one GPU-heavy new step)
#   Phase 3  layer_analysis.py — SEAL Fig. 3 replication + separation metrics
#            >>> ANALYSIS_ONLY=1 stops here (CPU results, team checkpoint) <<<
#   Phase 4  Per-layer vectors (CPU, vector_generation.py)
#   Phase 5  Baseline GSM8K on a seeded random sample (vLLM, fast)
#   Phase 6  Steered GSM8K at each layer in EVAL_LAYERS (HF generate — SLOW:
#            expect ~1.5-3h per layer at n=300 on a 24GB card)
#
# Usage:
#   bash scripts/layer_sweep_gsm.sh                    # everything
#   ANALYSIS_ONLY=1 bash scripts/layer_sweep_gsm.sh    # stop after Fig-3 analysis
#   EVAL_LAYERS="5 20" GSM_MAX_EXAMPLES=100 bash scripts/layer_sweep_gsm.sh  # smoke
#
# Re-running is cheap: completed phases whose outputs exist are skipped.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# --- Config (override via env) ---
if [[ -f .env ]]; then set -a; source .env; set +a; fi
: "${GPU:=0}"
: "${HF_HOME:=/workspace/hf_cache}"
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${MAX_TOKENS:=10000}"
: "${MATH_TRAIN_CAP:=3000}"
: "${VEC_SAMPLES:=500}"
: "${STEER_COEF:=-1.0}"
: "${BATCH_SIZE:=8}"                       # steered eval; 8 was OOM-safe on a 4090
: "${ANALYSIS_LAYERS:=1 5 10 15 20 25 28}" # paper's Fig-3/Fig-6 grid
: "${EVAL_LAYERS:=5 15 20 25}"             # early / mid / default / plateau-check
: "${GSM_MAX_EXAMPLES:=300}"               # seeded random sample of official test
: "${SAMPLE_SEED:=0}"
: "${ANALYSIS_ONLY:=0}"

export HF_HOME CUDA_VISIBLE_DEVICES="$GPU"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL_TAG="$(basename "$MODEL")"
VEC_BASE="results/MATH_train/${MODEL_TAG}/baseline_${MAX_TOKENS}"

echo "=================================================================="
echo " MODEL=$MODEL  GPU=$GPU"
echo " analysis layers: $ANALYSIS_LAYERS"
echo " eval layers:     $EVAL_LAYERS  (coef $STEER_COEF, GSM n=$GSM_MAX_EXAMPLES seed=$SAMPLE_SEED)"
echo " ANALYSIS_ONLY=$ANALYSIS_ONLY"
echo "=================================================================="

# ------------------------------------------------------------------ #
# Phase 1: MATH-train baseline traces (identical to runpod_gsm.sh)
# ------------------------------------------------------------------ #
if [[ -f "${VEC_BASE}/math_eval.jsonl" ]]; then
    echo "[phase1] MATH-train baseline traces already exist, skipping."
else
    echo "[phase1] Generating MATH-train baseline traces (cap=$MATH_TRAIN_CAP)..."
    python eval_MATH_vllm.py \
        --model_name_or_path "$MODEL" \
        --save_dir "$VEC_BASE" \
        --max_tokens "$MAX_TOKENS" \
        --max_examples "$MATH_TRAIN_CAP" \
        --use_chat_format \
        --dataset "MATH_train" \
        --remove_bos
fi

# ------------------------------------------------------------------ #
# Phase 2: hidden states at ALL analysis layers.
# NOTE: this intentionally does NOT reuse the runpod_gsm.sh extraction,
# which was run with --keep_layers 20 and lacks the other layers.
# Rough size: 1000 traces x ~50 steps x 1536 dims x fp32 x 7 layers ~= 2 GB/pool.
# ------------------------------------------------------------------ #
for TYPE in incorrect correct; do
    HDIR="${VEC_BASE}/hidden_${TYPE}_0_${VEC_SAMPLES}"
    if [[ -f "${HDIR}/hidden.pt" ]]; then
        echo "[phase2] ${TYPE} hidden states already exist, skipping."
        echo "         (verify they contain layers: $ANALYSIS_LAYERS — a"
        echo "          --keep_layers 20 artifact will fail in phase 3)"
    else
        echo "[phase2] Extracting ${TYPE} hidden states (layers: $ANALYSIS_LAYERS)..."
        python hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path data/MATH/train.jsonl \
            --data_dir "$VEC_BASE" \
            --type "$TYPE" --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers $ANALYSIS_LAYERS
    fi
done

# ------------------------------------------------------------------ #
# Phase 3: Fig-3 replication + separation metrics (CPU)
# ------------------------------------------------------------------ #
echo "[phase3] Layer separation analysis..."
python layer_analysis.py \
    --data_dir "$VEC_BASE" \
    --prefixs "correct_0_${VEC_SAMPLES}" "incorrect_0_${VEC_SAMPLES}" \
    --layers $ANALYSIS_LAYERS

if [[ "$ANALYSIS_ONLY" == "1" ]]; then
    echo "ANALYSIS_ONLY=1 — stopping after Fig-3 analysis."
    echo "  outputs: ${VEC_BASE}/layer_analysis/"
    exit 0
fi

# ------------------------------------------------------------------ #
# Phase 4: per-layer steering vectors (CPU)
# ------------------------------------------------------------------ #
echo "[phase4] Building vectors at layers: $EVAL_LAYERS ..."
python vector_generation.py \
    --data_dir "$VEC_BASE" \
    --prefixs "correct_0_${VEC_SAMPLES}" "incorrect_0_${VEC_SAMPLES}" \
    --layers $EVAL_LAYERS \
    --save_prefix "${VEC_SAMPLES}_${VEC_SAMPLES}"

# ------------------------------------------------------------------ #
# Phase 5: baseline GSM8K on the seeded random sample (vLLM)
# ------------------------------------------------------------------ #
echo "[phase5] Baseline GSM8K eval (seeded sample)..."
python eval_MATH_vllm.py \
    --model_name_or_path "$MODEL" \
    --save_dir "results/GSM/${MODEL_TAG}/layer_sweep_baseline_${MAX_TOKENS}" \
    --max_tokens "$MAX_TOKENS" \
    --use_chat_format \
    --dataset "GSM" \
    --remove_bos \
    --random_sample --sample_seed "$SAMPLE_SEED" \
    --max_examples "$GSM_MAX_EXAMPLES"

# ------------------------------------------------------------------ #
# Phase 6: steered GSM8K per layer (HF generate — the slow part)
# ------------------------------------------------------------------ #
for L in $EVAL_LAYERS; do
    VEC_FILE="${VEC_BASE}/vector_${VEC_SAMPLES}_${VEC_SAMPLES}/layer_${L}_transition_reflection_steervec.pt"
    echo "[phase6] Steered GSM8K eval — layer $L (coef $STEER_COEF)..."
    python eval_MATH_steering.py \
        --model_name_or_path "$MODEL" \
        --save_dir "results/GSM/${MODEL_TAG}/layer_sweep_${MAX_TOKENS}" \
        --max_tokens "$MAX_TOKENS" \
        --use_chat_format \
        --batch_size "$BATCH_SIZE" \
        --dataset "GSM" \
        --remove_bos \
        --steering \
        --steering_vector "$VEC_FILE" \
        --steering_layer "$L" \
        --steering_coef "$STEER_COEF" \
        --random_sample --sample_seed "$SAMPLE_SEED" \
        --max_examples "$GSM_MAX_EXAMPLES"
done

echo "=================================================================="
echo " DONE."
echo "   Fig-3 analysis : ${VEC_BASE}/layer_analysis/"
echo "   baseline       : results/GSM/${MODEL_TAG}/layer_sweep_baseline_${MAX_TOKENS}/"
echo "   steered        : results/GSM/${MODEL_TAG}/layer_sweep_${MAX_TOKENS}/  (one dir per layer vector)"
echo "=================================================================="
grep -r '"acc"' "results/GSM/${MODEL_TAG}/layer_sweep"* 2>/dev/null || true
