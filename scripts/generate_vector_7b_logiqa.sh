#!/usr/bin/env bash
# Build the LogiQA (logic) steering vector for DeepSeek-R1-Distill-Qwen-7B.
# Mirrors scripts/generate_vector_7b_apps.sh and scripts/generate_vector_logiqa.sh:
#   gen_logiqa_vllm.py (vLLM)  ->  hidden_analysis.py --keywords logic  ->  vector_generation.py
#
#   TARGET=20 MAX_EXAMPLES=80 bash scripts/generate_vector_7b_logiqa.sh 0   # smoke
#   bash scripts/generate_vector_7b_logiqa.sh 0                             # full 500+500
#
# Eval the result with scripts/eval_7b_s_logic.sh (VECTOR= the .pt this writes).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
: "${MAX_TOKENS:=10000}"
: "${TARGET:=500}"                 # traces per class (correct/incorrect)
: "${KEYWORDS:=logic}"
: "${SAVE_TAG:=baseline_10000}"
: "${MAX_EXAMPLES:=}"              # empty = all CJK-filtered LogiQA2.0 train rows
TAG=$(basename "$MODEL")
DIR="results/train_logic/${TAG}/${SAVE_TAG}"

if [[ "$KEYWORDS" == "logic" ]]; then
    python -c "
from hidden_analysis import KEYWORD_SETS
assert 'logic' in KEYWORD_SETS, 'KEYWORD_SETS[\"logic\"] missing in hidden_analysis.py'
"
fi

MAX_EX_ARGS=()
if [[ -n "$MAX_EXAMPLES" ]]; then
    MAX_EX_ARGS=(--max_examples "$MAX_EXAMPLES")
fi

if [[ -f "$DIR/math_eval.jsonl" && -f "$DIR/data.jsonl" ]]; then
    echo "[1/4] $DIR/math_eval.jsonl + data.jsonl already exist -- skipping generation"
    echo "      (delete these files to force a re-run)"
    python - "$DIR/math_eval.jsonl" <<'PY'
import json, sys
bad = tot = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    gen = r["model_generation"][0]
    tot += 1
    if "</think>" not in gen and r["all_eval"][0]:
        bad += 1
if bad:
    raise SystemExit(
        f"[guard] {bad}/{tot} reused traces are labelled CORRECT despite never closing "
        f"</think>. Re-grade before building the vector."
    )
print(f"[guard] {tot} reused traces pass the grading check.")
PY
else
    echo "[1/4] vLLM generate + score LogiQA train (CJK-filtered, max_tokens=${MAX_TOKENS}) ..."
    CUDA_VISIBLE_DEVICES=$gpu python -u gen_logiqa_vllm.py \
        --model_name_or_path "$MODEL" --save_dir "$DIR" \
        --max_tokens "$MAX_TOKENS" --split train --filter_cjk "${MAX_EX_ARGS[@]}" \
        --use_chat_format --remove_bos
fi

echo "[2/4] hidden states — incorrect (layer 20, keywords=${KEYWORDS}) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type incorrect --start 0 --sample "$TARGET" --keep_layers 20 --keywords "$KEYWORDS"

echo "[3/4] hidden states — correct (layer 20, keywords=${KEYWORDS}) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type correct --start 0 --sample "$TARGET" --keep_layers 20 --keywords "$KEYWORDS"

echo "[4/4] build steering vector (layer 20) ..."
python -u vector_generation.py \
    --data_dir "$DIR" --prefixs "correct_0_${TARGET}" "incorrect_0_${TARGET}" \
    --layers 20 --save_prefix "${KEYWORDS}_${TARGET}_${TARGET}" --overwrite

echo "== done: $DIR/vector_${KEYWORDS}_${TARGET}_${TARGET}/layer_20_transition_reflection_steervec.pt =="
echo "   (SEAL convention: vector = H_RT - H_E; apply with coef -1.0 in the eval)"
echo "   VECTOR=$DIR/vector_${KEYWORDS}_${TARGET}_${TARGET}/layer_20_transition_reflection_steervec.pt \\"
echo "     bash scripts/eval_7b_s_logic.sh"
