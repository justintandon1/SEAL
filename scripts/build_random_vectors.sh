#!/usr/bin/env bash
# Build the R_iso control draws (Control A). CPU only, a few seconds.
#
# Each seed is screened against the five SEAL vectors AND against every earlier
# seed, so the draws are verified mutually near-orthogonal as they are built.
# Rationale: docs/random_vector_build_plan.html
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=${SEEDS:-"1 2 3"}
OUT_DIR=${OUT_DIR:-results/control}
LIKE=${LIKE:-results/general/S_combo.pt}
PREFIX=${PREFIX:-R_iso}

S_MATH=results/results_for_math_vectors/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt

REFS=(
  --check-against "S_math=$S_MATH"
  --check-against "S_code=vectors/apps_v_code.pt"
  --check-against "S_logic=vectors/logiqa_v_logic.pt"
  --check-against "S_general=results/general/S_general_math_apps_logic_phase1.pt"
  --check-against "S_combo=results/general/S_combo.pt"
)

for seed in $SEEDS; do
  out="$OUT_DIR/${PREFIX}_seed${seed}.pt"
  prior=()
  for earlier in $SEEDS; do
    [[ "$earlier" == "$seed" ]] && break
    prior+=(--check-against "${PREFIX}_seed${earlier}=$OUT_DIR/${PREFIX}_seed${earlier}.pt")
  done

  python build_random_vector.py \
    --seed "$seed" \
    --like "$LIKE" \
    "${REFS[@]}" \
    ${prior[@]+"${prior[@]}"} \
    --out "$out"
  echo
done

echo "Built: $OUT_DIR/${PREFIX}_seed{$(echo $SEEDS | tr ' ' ',')}.pt"
