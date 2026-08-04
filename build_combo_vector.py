"""Build S_combo, the weighted-combination baseline for S_general.

``S_combo`` answers the geometry half of the research question: is a pooled
general calibration vector just the sum of the per-domain vectors, or a
distinct direction? It is built from the finished domain vectors only — no
``hidden.pt`` and no GPU.

    S_combo = normalize_scale( a·Ŝ_math + b·Ŝ_code + c·Ŝ_logic )

with each ``Ŝ`` unit-normalized first and ``a = b = c = 1`` by default.

Why equal weights on unit-normalized inputs
    Measured by ``scripts/vector_geometry.py`` on the shipped vectors: the
    three are close to rank-1 (singular values [1.613, 0.541, 0.325]; the top
    direction carries 86.7% of the variance), so reweighting cannot rotate the
    sum much — over a 4x weight range the direction stays within cos 0.985 of
    equal weights. Grid-searching the weights would spend GPU budget resolving
    differences of ~0.015 in cosine. Equal weights is also the correct
    comparison object: a domain-balanced ``S_general`` equals the average of
    the domain vectors, so equal weights tests *pooling vs. combining* rather
    than *tuned vs. untuned*.

    Unit-normalizing is free here — the source norms are within 8% of each
    other, so cos(raw sum, unit sum) = 0.99996 — but it is the more defensible
    choice to write up, and it keeps the build correct if a future vector
    arrives with a very different norm.

Why the output is rescaled
    ``eval_*_steering.py`` applies ``steer_coef * steer_vec`` with
    ``steer_coef = -1.0``. The unit sum has norm ~2.8 against ~56 for the
    domain vectors, so saving it unscaled would make S_combo a ~20x weaker
    intervention at the same coefficient and it would appear inert. The
    default ``--norm mean-source`` rescales to the mean of the input norms so
    that ``coef = -1.0`` is a like-for-like intervention. Scaling never changes
    the direction, and the factor is recorded in the meta.

Sign convention
    Inputs and output are ``mean(check ∪ switch) − mean(other)``, applied by
    ADDING ``coef * S`` with ``coef = -1.0`` — same as
    ``build_general_vector.py`` and ``vector_generation.py``. A vector built
    the other way round would cancel rather than combine when summed, so the
    build aborts if the inputs disagree on sign.

Examples::

    # Default: equal weights over the three domain vectors.
    python build_combo_vector.py --out results/general/S_combo.pt

    # Compare against S_general once it exists.
    python build_combo_vector.py --out results/general/S_combo.pt \\
        --compare general=results/general/S_general_math_apps_logic_phase1.pt

    # Non-equal weights (ablation only — see the note above).
    python build_combo_vector.py --out results/general/S_combo_w.pt \\
        --weight code=2 --weight logic=0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_general_vector import _parse_domain, cosine  # noqa: E402
from scripts.vector_geometry import (  # noqa: E402
    DEFAULT_VECTORS,
    load_vectors,
    span_structure,
    unit,
)


def parse_weights(specs: Sequence[str], names: Sequence[str]) -> Dict[str, float]:
    """Parse ``--weight name=value`` into a full weight map defaulting to 1.0.

    Args:
        specs: ``name=value`` strings. Names not listed default to 1.0.
        names: Every vector name, used to validate and to fill defaults.

    Returns:
        One weight per name, in the order of ``names``.

    Raises:
        ValueError: If a spec names an unknown vector or a non-numeric weight.
    """
    weights = {name: 1.0 for name in names}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--weight expected name=value, got: {spec!r}")
        name, value = spec.split("=", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(
                f"--weight names unknown vector {name!r}; have {list(names)}"
            )
        try:
            weights[name] = float(value)
        except ValueError as exc:
            raise ValueError(
                f"--weight {name}: expected a number, got {value!r}"
            ) from exc
    return weights


def check_sign_convention(vectors: Dict[str, torch.Tensor]) -> None:
    """Abort if any pair of inputs points in opposing directions.

    Every input must be ``mean(check ∪ switch) − mean(other)``. A vector built
    with the opposite orientation shows up as a negative cosine against the
    others, and summing it would cancel signal instead of accumulating it.

    Raises:
        ValueError: If any off-diagonal cosine is negative.
    """
    names = list(vectors)
    bad: List[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = cosine(vectors[a], vectors[b])
            if c < 0:
                bad.append(f"{a} vs {b}: cos={c:+.4f}")
    if bad:
        raise ValueError(
            "input vectors disagree on sign convention, so summing them would "
            "cancel rather than combine:\n  " + "\n  ".join(bad) + "\n"
            "Every input must be mean(check ∪ switch) − mean(other). Negate the "
            "offending vector, or check which script produced it."
        )


def build_combo(
    vectors: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    norm_mode: str,
) -> tuple[torch.Tensor, Dict[str, object]]:
    """Build the weighted combination and rescale it for evaluation.

    Args:
        vectors: Domain vectors, all the same hidden dimension.
        weights: One weight per vector name.
        norm_mode: ``mean-source`` (default) rescales to the mean input norm so
            ``coef = -1.0`` matches the domain vectors; ``unit`` gives norm 1;
            ``none`` leaves the weighted sum unscaled; a float sets the norm
            explicitly. Scaling never changes the direction.

    Returns:
        The scaled vector and a dict describing how it was scaled.

    Raises:
        ValueError: If ``norm_mode`` is not recognized.
    """
    unit_vectors = unit(vectors)
    combo = sum(weights[name] * v for name, v in unit_vectors.items())
    pre_norm = float(combo.norm())

    source_norms = {name: float(v.norm()) for name, v in vectors.items()}
    if norm_mode == "mean-source":
        target = sum(source_norms.values()) / len(source_norms)
    elif norm_mode == "unit":
        target = 1.0
    elif norm_mode == "none":
        target = pre_norm
    else:
        try:
            target = float(norm_mode)
        except ValueError as exc:
            raise ValueError(
                f"--norm expected 'mean-source', 'unit', 'none' or a number, "
                f"got {norm_mode!r}"
            ) from exc

    scaled = combo * (target / pre_norm)
    return scaled, {
        "mode": norm_mode,
        "weighted_sum_norm": round(pre_norm, 6),
        "target_norm": round(target, 6),
        "scale_factor": round(target / pre_norm, 6),
        "source_norms": {k: round(v, 4) for k, v in source_norms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build S_combo = weighted sum of unit-normalized domain "
        "calibration vectors."
    )
    parser.add_argument(
        "--vector",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Domain vector to combine, repeatable. Defaults to the three "
        "vectors tracked in the repo (math, code, logic).",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Weight for one vector; unlisted vectors default to 1.0. Equal "
        "weights is the recommended primary — see the module docstring.",
    )
    parser.add_argument(
        "--norm",
        default="mean-source",
        help="Output magnitude: 'mean-source' (default, matches the domain "
        "vectors so coef -1.0 is like-for-like), 'unit', 'none', or a number.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Extra vector to report cosine against, repeatable. Use this for "
        "S_general once it is built.",
    )
    parser.add_argument("--out", required=True, help="Output .pt path.")
    args = parser.parse_args()

    vectors = load_vectors(args.vector or DEFAULT_VECTORS)
    check_sign_convention(vectors)
    weights = parse_weights(args.weight, list(vectors))
    combo, scaling = build_combo(vectors, weights, args.norm)

    print("=" * 66)
    print(" S_combo")
    print("=" * 66)
    for name, v in vectors.items():
        print(f"  {name:<8} weight={weights[name]:<6g} norm={v.norm():.3f}")
    print(f"\n  weighted sum norm     {scaling['weighted_sum_norm']:.4f}")
    print(
        f"  rescaled to           {scaling['target_norm']:.4f}"
        f"  (x{scaling['scale_factor']:.4f}, mode={scaling['mode']})"
    )

    print("\n  cosine with each source vector")
    cos_sources = {name: cosine(combo, v) for name, v in vectors.items()}
    for name, c in cos_sources.items():
        print(f"    {name:<8} {c:+.4f}")

    cos_extra: Dict[str, float] = {}
    if args.compare:
        print("\n  cosine with comparison vectors")
        for spec in args.compare:
            name, paths = _parse_domain(spec)
            other = torch.load(
                paths[0], map_location="cpu", weights_only=True
            ).float().flatten()
            cos_extra[name] = cosine(combo, other)
            print(f"    {name:<8} {cos_extra[name]:+.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(combo, args.out)

    meta = {
        "name": os.path.splitext(os.path.basename(args.out))[0],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "recipe": "S_combo = scale * sum_d weight_d * (S_d / ||S_d||)",
        "sign_convention": (
            "vector = mean(check ∪ switch) − mean(other); apply by ADDING "
            "coef * vector with coef = -1.0 (same as build_general_vector.py)"
        ),
        "apply_coef": -1.0,
        "weights": weights,
        "sources": {
            name: spec.split("=", 1)[1]
            for name, spec in zip(vectors, args.vector or DEFAULT_VECTORS)
        },
        "scaling": scaling,
        "vector_dim": int(combo.shape[0]),
        "vector_norm": round(float(combo.norm()), 6),
        "cosine_with_sources": {k: round(v, 6) for k, v in cos_sources.items()},
        "cosine_with_comparisons": {
            k: round(v, 6) for k, v in cos_extra.items()
        },
        "span_of_sources": span_structure(vectors),
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    print("\n" + "=" * 66)
    print(f"  vector : {args.out}")
    print(f"  meta   : {meta_path}")
    print(f"  apply  : coef -1.0 (same layer as the source vectors)")
    print("=" * 66)


if __name__ == "__main__":
    main()
