#!/usr/bin/env bash
# Control A on MATH-500: three R_iso draws, ~3-4.5 GPU-h total.
#
# Scope is deliberately one benchmark. The full 3x6 grid would be 18 runs, on
# the order of the entire ~38 GPU-h already spent on the study. MATH-500 carries
# it alone because the control belongs where the effect is strongest (+15.0, so
# a pass generalizes downward), where the truncation mechanism is most visible
# (32.8% baseline cap rate), and where an arm is cheapest. Expand only if a seed
# comes back hot -- and then extend that seed, not all three.
#
# Rationale: docs/random_vector_build_plan.html
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU_INDEX=${1:-0}
SEEDS=${SEEDS:-"1 2 3"}
VEC_DIR=${VEC_DIR:-results/control}
PREFIX=${PREFIX:-R_iso}
export RESULT_ROOT=${RESULT_ROOT:-results/results_for_control_vectors}

# The existing MATH-500 baseline is reused rather than regenerated. That is only
# valid while these match the arm it is paired against -- the exact paired
# McNemar test depends on identical problem ordering, and a mismatch produces a
# number that looks fine and means nothing.
export RUN_BASELINE=0
export DATASETS=math
export SAMPLE_SEED=${SAMPLE_SEED:-42}
export MATH_MAX_EXAMPLES=${MATH_MAX_EXAMPLES:-500}
export MAX_TOKENS=${MAX_TOKENS:-10000}

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
