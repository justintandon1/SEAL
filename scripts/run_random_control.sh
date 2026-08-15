#!/usr/bin/env bash
# Control A: three R_iso draws. ~11 GPU-h per arm per benchmark.
#
#   bash scripts/run_random_control.sh 0                    # MATH-500 (default)
#   DATASETS=logiqa bash scripts/run_random_control.sh 0     # LogiQA clean-500
#
# The build plan scopes this to MATH-500 alone: the full 3x6 grid would be 18
# runs, on the order of the entire ~38 GPU-h already spent on the study, and
# MATH-500 carries it because the control belongs where the effect is strongest
# (+15.0, so a pass generalizes downward), where the truncation mechanism is
# most visible (32.8% baseline cap rate), and where an arm is cheapest. The
# plan's rule is: expand only if a seed comes back hot -- and then extend that
# seed, not all three.
#
# LogiQA is wired up here so that expansion is one env var rather than a
# rebuild. Note before using it that the LogiQA effect is much smaller than
# MATH's -- baseline 0.258 against 0.284-0.298 for the SEAL vectors, i.e. 13-20
# problems out of 500 -- so a control arm there has far less room to distinguish
# "does nothing" from "reproduces the effect" than it does on MATH.
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
