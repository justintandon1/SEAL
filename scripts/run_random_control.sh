#!/usr/bin/env bash
# Control A: R_iso draws. Budget ~7.4 GPU-h per MATH-500 arm, measured against
# the seed-1 run of 2026-08-15 (16:22-23:50 on an A100 PCIe). APPS costs several
# times that; see the build plan's budget table before launching one.
#
# The plan was re-split on 2026-08-17 into two phases:
#
#   PHASE 1 -- seed 1 across MATH-500, LogiQA and APPS. MATH-500 is done.
#     SEEDS=1 DATASETS=logiqa bash scripts/run_random_control.sh 0
#     SEEDS=1 DATASETS=apps   bash scripts/run_random_control.sh 0
#
#   PHASE 2 -- seeds 2 and 3, MATH-500 first.
#     SEEDS="2 3" DATASETS=math bash scripts/run_random_control.sh 0
#
# Phase 2 is required, not optional. The rank-sum test over 5 SEAL arms and n
# random arms has floor p = 1/C(5+n, n), so at n=1 it is 0.167 one-sided and
# 0.333 two-sided: no phase-1 result reaches significance, on any benchmark.
# Adding benchmarks does not help -- the floor is set by the number of draws.
# Phase 1 supports the per-arm paired McNemar and nothing beyond it.
#
# Note before running LogiQA that its effect is much smaller than MATH's --
# baseline 0.258 against 0.284-0.298 for the SEAL vectors, i.e. 13-20 problems
# out of 500 -- so a control arm there has far less room to distinguish "does
# nothing" from "reproduces the effect" than it does on MATH.
#
# Always tee to a log on disk: the Aug 15 APPS baseline died at startup and left
# no record of why. See docs/vast_runbook.md.
#
# Rationale: docs/random_vector_build_plan.html
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU_INDEX=${1:-0}
SEEDS=${SEEDS:-"1 2 3"}
VEC_DIR=${VEC_DIR:-results/control}
PREFIX=${PREFIX:-R_iso}
export RESULT_ROOT=${RESULT_ROOT:-results/results_for_control_vectors}

# The existing baselines are reused rather than regenerated. That is only valid
# while these match the arm they are paired against -- the exact paired McNemar
# test depends on identical problem ordering, and a mismatch produces a number
# that looks fine and means nothing.
export RUN_BASELINE=0
export DATASETS=${DATASETS:-math}
export SAMPLE_SEED=${SAMPLE_SEED:-42}
export MATH_MAX_EXAMPLES=${MATH_MAX_EXAMPLES:-500}
export MAX_TOKENS=${MAX_TOKENS:-10000}

# LogiQA settings, used only when DATASETS includes logiqa. These reproduce what
# every existing LogiQA arm ran: the contamination-free explicit selection (not
# the seeded sample), English-only pool, 10k budget, batch 25. Verified against
# results_for_logic_vectors/LogiQA/baseline -- same 500 problems, same order, so
# RUN_BASELINE=0 is sound for LogiQA too.
export LOGIQA_ENGLISH_ONLY=${LOGIQA_ENGLISH_ONLY:-1}
export LOGIQA_SELECTION=${LOGIQA_SELECTION:-data/LogiQA/eval_rand42_500_clean.json}
export LOGIQA_BATCH_SIZE=${LOGIQA_BATCH_SIZE:-25}

# Refuse to burn GPU hours on vectors that have not cleared the CPU checklist.
echo ">>> Verifying control vectors before any GPU time"
python scripts/verify_random_vectors.py --dir "$VEC_DIR" --prefix "$PREFIX"
echo

for seed in $SEEDS; do
  vector="$VEC_DIR/${PREFIX}_seed${seed}.pt"
  echo ">>> ${PREFIX}_seed${seed}  ($vector)"
  scripts/eval_steering_vector_benchmarks.sh "$vector" "${PREFIX}_seed${seed}" "$GPU_INDEX"
  echo
done

echo ">>> All arms finished. Results under $RESULT_ROOT"
echo ">>> Next: python scripts/report_stats.py"
