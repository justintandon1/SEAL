#!/usr/bin/env bash
# Run one steered arm across every GPU on the box, on MATH-500 or LogiQA.
#
# The arm is data-parallel over batches: N independent processes, one pinned to
# each GPU, no communication between them. There are no gradients here, so
# torchrun/DDP/FSDP would add coordination and failure modes and buy nothing.
# Batches are dealt out round-robin as whole units, so the batch an example
# lands in -- and therefore its padding and its numerics -- is identical to a
# single-GPU run.
#
# Every batch is checkpointed, so a killed process, an OOM, or a spot reclaim
# costs one batch. Re-running resumes; it never redoes finished work.
#
#   DATASET=logiqa scripts/run_arm_multigpu.sh results/control/R_iso_seed1.pt R_iso_seed1
#   DATASET=math   scripts/run_arm_multigpu.sh results/control/R_iso_seed2.pt R_iso_seed2
#
# Env: DATASET NUM_GPUS MAX_ATTEMPTS RESULT_ROOT MODEL LAYER COEF MAX_TOKENS
#      BATCH_SIZE SAMPLE_SEED MAX_EXAMPLES LOGIQA_SELECTION
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: DATASET=math|logiqa $0 VECTOR_PATH|baseline [VECTOR_NAME]" >&2
  exit 2
fi

VECTOR_PATH=$1
VECTOR_NAME=${2:-$(basename "${VECTOR_PATH%.pt}")}

DATASET=${DATASET:-math}
MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
LAYER=${LAYER:-20}
COEF=${COEF:--1.0}
MAX_TOKENS=${MAX_TOKENS:-10000}
BATCH_SIZE=${BATCH_SIZE:-25}
SAMPLE_SEED=${SAMPLE_SEED:-42}
MAX_EXAMPLES=${MAX_EXAMPLES:-500}
RESULT_ROOT=${RESULT_ROOT:-results/results_for_control_vectors}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l | tr -d ' ')}
# Resume is free, so a transient failure is worth retrying before giving up.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

# LogiQA is scored on the contamination-free explicit selection, not the seeded
# sample -- that is what every existing LogiQA arm and the reusable baseline
# used. Changing it silently breaks the pairing with those arms.
LOGIQA_SELECTION=${LOGIQA_SELECTION:-data/LogiQA/eval_rand42_500_clean.json}

case "$DATASET" in
  math)
    DATASET_ARG=MATH500
    BENCH_DIR=MATH500
    DATASET_ARGS=(--dataset MATH500 --random_sample --sample_seed "$SAMPLE_SEED" --max_examples "$MAX_EXAMPLES")
    ;;
  logiqa)
    DATASET_ARG=LogiQA
    BENCH_DIR=LogiQA
    if [[ ! -f "$LOGIQA_SELECTION" ]]; then
      echo "LOGIQA_SELECTION not found: $LOGIQA_SELECTION" >&2
      exit 2
    fi
    DATASET_ARGS=(--dataset LogiQA --logiqa_english_only --logiqa_eval_selection "$LOGIQA_SELECTION")
    ;;
  *)
    echo "Unknown DATASET '$DATASET'. Use math or logiqa." >&2
    exit 2
    ;;
esac

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No GPUs visible to nvidia-smi." >&2
  exit 2
fi

ARGS=(
  --model_name_or_path "$MODEL"
  --max_tokens "$MAX_TOKENS"
  --use_chat_format
  --batch_size "$BATCH_SIZE"
  --remove_bos
  "${DATASET_ARGS[@]}"
  --num_shards "$NUM_GPUS"
)

if [[ "$VECTOR_PATH" == "baseline" ]]; then
  ARGS+=(--save_dir "$RESULT_ROOT/$BENCH_DIR/baseline")
else
  if [[ ! -f "$VECTOR_PATH" ]]; then
    echo "Steering vector not found: $VECTOR_PATH" >&2
    exit 2
  fi
  if [[ ! "$VECTOR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid VECTOR_NAME '$VECTOR_NAME'; use letters, numbers, dots, dashes, or underscores" >&2
    exit 2
  fi
  ARGS+=(
    --save_dir "$RESULT_ROOT/$BENCH_DIR/$VECTOR_NAME"
    --steering
    --steering_vector "$VECTOR_PATH"
    --steering_layer "$LAYER"
    --steering_coef "$COEF"
  )
fi

# Ask the eval for the directory it will actually write to, rather than
# reimplementing its nesting rules here and drifting from them later.
SAVE_DIR=$(python eval_MATH_steering.py "${ARGS[@]}" --print_save_dir)
LOG_DIR="$SAVE_DIR/logs"
mkdir -p "$LOG_DIR"

echo ">>> arm      : $VECTOR_NAME"
echo ">>> dataset  : $DATASET_ARG"
echo ">>> gpus     : $NUM_GPUS"
echo ">>> save_dir : $SAVE_DIR"
echo ">>> logs     : $LOG_DIR/rank<N>.log"
echo

# Warm the model cache serially. N processes racing to download the same 7 GB of
# fp32 weights is how you get a corrupt cache and N crashed shards. LogiQA reads
# from data/LogiQA/ on disk, so only MATH needs the dataset warmed too.
echo ">>> Warming caches"
python - "$MODEL" "$DATASET" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
if sys.argv[2] == "math":
    from datasets import load_dataset
    load_dataset("HuggingFaceH4/MATH-500", split="test")
print("caches ready")
PY
echo

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo ">>> Attempt $attempt/$MAX_ATTEMPTS -- launching $NUM_GPUS shards"
  pids=()
  ranks=()
  for rank in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES="$rank" python eval_MATH_steering.py \
      "${ARGS[@]}" --shard_rank "$rank" \
      >"$LOG_DIR/rank${rank}.log" 2>&1 &
    pids+=($!)
    ranks+=("$rank")
  done

  failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "!!! rank ${ranks[$i]} failed -- see $LOG_DIR/rank${ranks[$i]}.log" >&2
      failed=1
    fi
  done

  if [[ "$failed" -eq 0 ]]; then
    echo ">>> All shards finished"
    break
  fi
  if [[ "$attempt" -eq "$MAX_ATTEMPTS" ]]; then
    echo "!!! Giving up after $MAX_ATTEMPTS attempts. Finished batches are" >&2
    echo "!!! checkpointed -- re-running this script resumes from them." >&2
    exit 1
  fi
  echo ">>> Retrying; completed batches will be skipped"
  sleep 10
done

echo
echo ">>> Merging shards and scoring"
python scripts/merge_shards.py --save_dir "$SAVE_DIR" --dataset "$DATASET_ARG"
