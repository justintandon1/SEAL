"""Geometry report for the SEAL calibration vectors.

Answers, from the ``.pt`` files alone, the questions that decide how
``S_combo`` should be built:

1. Are the domain vectors comparable in scale? (norms)
2. How aligned are they? (pairwise cosine matrix)
3. Is their span effectively low-rank? (singular values)
4. Does unit-normalizing before summing change the combo direction?
5. Do the combination weights matter? (weight-sensitivity sweep)
6. Is the alignment inflated by the dominant activation direction?
   (``--hidden`` diagnostic; needs ``hidden.pt``)

This is pure linear algebra on the stored vectors — no model is loaded and no
text is generated. It runs on CPU in about a second, so it is cheap to re-run
after any vector rebuild.

Sign convention: every vector is expected to be
``mean(check ∪ switch) − mean(other)``, applied by ADDING ``coef * S`` with
``coef = -1.0`` (see ``build_general_vector.build_vector``). Vectors built the
other way round would show up as negative cosines against the rest; the report
flags that rather than silently averaging opposed directions.

Examples::

    # Default: the three domain vectors shipped in the repo.
    python scripts/vector_geometry.py

    # Add S_general once it exists.
    python scripts/vector_geometry.py \\
        --vector general=results/general/S_general_math_apps_logic_phase1.pt

    # Include the dominant-direction diagnostic (needs hidden.pt on disk).
    python scripts/vector_geometry.py \\
        --hidden math=data/MATH/hidden_correct_0_500/hidden.pt,data/MATH/hidden_incorrect_0_500/hidden.pt \\
        --hidden apps=data/APPS/hidden_correct_0_500/hidden.pt,data/APPS/hidden_incorrect_0_500/hidden.pt \\
        --hidden logic=data/LogiQA/hidden_correct_0_500/hidden.pt,data/LogiQA/hidden_incorrect_0_500/hidden.pt

    # Machine-readable output for the write-up.
    python scripts/vector_geometry.py --json results/general/vector_geometry.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_general_vector import _parse_domain, cosine, load_domain  # noqa: E402

# The three vectors tracked in the repo. Paths are relative to the repo root.
DEFAULT_VECTORS = [
    "math=results/results_for_math_vectors/MATH_train/"
    "DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/"
    "layer_20_transition_reflection_steervec.pt",
    "code=vectors/apps_v_code.pt",
    "logic=vectors/logiqa_v_logic.pt",
]

# Weight grid for the sensitivity sweep. The first vector's weight is pinned to
# 1.0 because the combo direction is scale-invariant, so only ratios matter.
DEFAULT_WEIGHT_GRID = (0.5, 1.0, 2.0)


def load_vectors(specs: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Load ``name=path`` vector specs into flat float32 tensors.

    Args:
        specs: ``name=path`` strings, one per vector.

    Returns:
        Mapping from name to a 1-D float32 tensor on CPU.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If two vectors have different hidden dimensions.
    """
    out: Dict[str, torch.Tensor] = {}
    for spec in specs:
        name, paths = _parse_domain(spec)
        path = paths[0]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"vector {name}: missing file at {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        out[name] = tensor.float().flatten()

    dims = {name: int(v.shape[0]) for name, v in out.items()}
    if len(set(dims.values())) > 1:
        raise ValueError(f"vectors have mismatched hidden dims: {dims}")
    return out


def unit(vectors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return each vector rescaled to unit L2 norm."""
    return {name: v / v.norm() for name, v in vectors.items()}


def cosine_matrix(
    vectors: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """Full pairwise cosine matrix, keyed ``[row][column]``."""
    names = list(vectors)
    return {
        a: {b: cosine(vectors[a], vectors[b]) for b in names} for a in names
    }


def span_structure(vectors: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Singular values of the stacked unit vectors, plus rank concentration.

    A large ``top_direction_share`` means the vectors are close to a single
    shared direction: reweighting a nearly rank-1 set cannot rotate the sum
    much, which is what makes the weight sweep below come out flat.

    Returns:
        ``singular_values`` and ``top_direction_share`` = σ₁² / Σσᵢ².
    """
    stacked = torch.stack([v for v in unit(vectors).values()])
    sv = torch.linalg.svdvals(stacked)
    return {
        "singular_values": [round(float(s), 4) for s in sv],
        "top_direction_share": round(float(sv[0] ** 2 / (sv**2).sum()), 4),
    }


def weighted_combo(
    unit_vectors: Dict[str, torch.Tensor], weights: Sequence[float]
) -> torch.Tensor:
    """Weighted sum of unit vectors, in the iteration order of the mapping."""
    parts = list(unit_vectors.values())
    if len(parts) != len(weights):
        raise ValueError(
            f"got {len(weights)} weights for {len(parts)} vectors"
        )
    return sum(w * v for w, v in zip(weights, parts))


def weight_sensitivity(
    unit_vectors: Dict[str, torch.Tensor],
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> Dict[str, object]:
    """How far reweighting moves the combo away from equal weights.

    The first vector's weight is pinned to 1.0 (the direction is scale
    invariant), and every remaining vector's weight sweeps ``grid``. A
    worst-case cosine close to 1.0 means grid-searching the combination
    weights cannot buy a materially different direction.

    Returns:
        ``grid``, every ``(weights, cosine)`` pair, and the ``worst_case``.
    """
    names = list(unit_vectors)
    equal = weighted_combo(unit_vectors, [1.0] * len(names))

    rows: List[Dict[str, object]] = []
    for tail in itertools.product(grid, repeat=len(names) - 1):
        weights = (1.0,) + tail
        rows.append(
            {
                "weights": dict(zip(names, weights)),
                "cosine_with_equal": round(
                    cosine(equal, weighted_combo(unit_vectors, weights)), 4
                ),
            }
        )

    worst = min(rows, key=lambda r: r["cosine_with_equal"])
    return {"grid": list(grid), "points": rows, "worst_case": worst}


def combo_variants(vectors: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Compare the raw-sum and unit-norm-sum combination directions.

    If ``cosine_raw_vs_unit`` is ~1.0, the two are the same direction and only
    one needs to be carried into the evaluation.
    """
    raw = sum(vectors.values())
    uni = sum(unit(vectors).values())
    equal = uni
    return {
        "raw_sum_norm": round(float(raw.norm()), 4),
        "unit_sum_norm": round(float(uni.norm()), 4),
        "cosine_raw_vs_unit": round(cosine(raw, uni), 6),
        "cosine_domain_vs_equal_weight_combo": {
            name: round(cosine(v, equal), 4) for name, v in vectors.items()
        },
    }


def dominant_direction(
    hidden_specs: Sequence[str],
    vectors: Dict[str, torch.Tensor],
    layer: int,
) -> Dict[str, object]:
    """Test whether alignment is inflated by the bulk activation direction.

    LLM activations are anisotropic: they cluster around a large common offset
    rather than the origin (Jorgensen et al., arXiv:2312.03813). Note the SEAL
    contrast ``mean(RT) − mean(E)`` already cancels any offset shared by *all*
    boundary rows exactly, so re-centring the activations cannot change the
    extracted vectors. What it can do is reveal that the vectors themselves lie
    along that bulk axis — in which case a high pairwise cosine reflects shared
    anisotropy rather than shared calibration structure.

    So this projects the bulk direction out of each vector and recomputes the
    geometry. If the cosines survive, the alignment is real.

    Args:
        hidden_specs: ``name=correct.pt,incorrect.pt`` per domain.
        vectors: The steering vectors to test.
        layer: Hidden layer to read, normally 20.

    Returns:
        Bulk-direction norm, its cosine with each vector, and the cosine matrix
        and span structure after the bulk direction is projected out.
    """
    totals: List[torch.Tensor] = []
    counts_by_domain: Dict[str, int] = {}
    for spec in hidden_specs:
        name, paths = _parse_domain(spec)
        pooled = load_domain(name, paths, layer)
        rows = torch.cat(
            [pooled["check"], pooled["switch"], pooled["other"]], dim=0
        ).float()
        counts_by_domain[name] = int(rows.shape[0])
        totals.append(rows.sum(dim=0))

    n_rows = sum(counts_by_domain.values())
    mu = torch.stack(totals).sum(dim=0) / n_rows
    mu_hat = mu / mu.norm()

    residual = {
        name: v - (v @ mu_hat) * mu_hat for name, v in vectors.items()
    }
    return {
        "boundary_rows_per_domain": counts_by_domain,
        "boundary_rows_total": n_rows,
        "bulk_direction_norm": round(float(mu.norm()), 4),
        "cosine_with_bulk_direction": {
            name: round(cosine(v, mu), 4) for name, v in vectors.items()
        },
        "cosine_matrix_bulk_removed": cosine_matrix(residual),
        "span_bulk_removed": span_structure(residual),
    }


def _print_matrix(matrix: Dict[str, Dict[str, float]]) -> None:
    """Print a cosine matrix as an aligned table."""
    names = list(matrix)
    width = max(len(n) for n in names) + 2
    print(" " * width + "".join(f"{n:>9}" for n in names))
    for a in names:
        cells = "".join(f"{matrix[a][b]:>+9.3f}" for b in names)
        print(f"{a:<{width}}{cells}")


def report(
    vectors: Dict[str, torch.Tensor],
    hidden_specs: Sequence[str],
    layer: int,
    grid: Sequence[float],
) -> Dict[str, object]:
    """Compute and print the full geometry report.

    Returns:
        The same numbers as a JSON-serializable dict.
    """
    out: Dict[str, object] = {"layer": layer}

    print("=" * 66)
    print(" Vector geometry")
    print("=" * 66)
    out["vectors"] = {
        name: {"dim": int(v.shape[0]), "norm": round(float(v.norm()), 4)}
        for name, v in vectors.items()
    }
    for name, info in out["vectors"].items():
        print(f"  {name:<8} dim={info['dim']:<6} norm={info['norm']:.3f}")

    print("\nPairwise cosine matrix")
    matrix = cosine_matrix(vectors)
    out["cosine_matrix"] = matrix
    _print_matrix(matrix)

    negatives = [
        (a, b)
        for a in matrix
        for b in matrix
        if a != b and matrix[a][b] < 0
    ]
    if negatives:
        print(
            "\n  WARNING: negative cosines present "
            f"({negatives[0][0]} vs {negatives[0][1]}). Vectors may disagree on "
            "sign convention; summing them would cancel rather than combine."
        )
    out["sign_convention_consistent"] = not negatives

    span = span_structure(vectors)
    out["span"] = span
    print(f"\nSpan structure (unit vectors)")
    print(f"  singular values        {span['singular_values']}")
    print(f"  top-direction share    {span['top_direction_share']:.3f}")

    variants = combo_variants(vectors)
    out["combo_variants"] = variants
    print("\nCombination directions")
    print(f"  ||raw sum||            {variants['raw_sum_norm']:.3f}")
    print(f"  ||unit sum||           {variants['unit_sum_norm']:.3f}")
    print(f"  cos(raw, unit)         {variants['cosine_raw_vs_unit']:.6f}")
    for name, c in variants["cosine_domain_vs_equal_weight_combo"].items():
        print(f"  cos({name}, equal combo){c:>+9.4f}")

    if len(vectors) > 1:
        sweep = weight_sensitivity(unit(vectors), grid)
        out["weight_sensitivity"] = sweep
        print(f"\nWeight sensitivity (first weight pinned to 1.0, grid={list(grid)})")
        for row in sweep["points"]:
            label = " ".join(f"{k}={v:g}" for k, v in row["weights"].items())
            print(f"  {label:<34} cos={row['cosine_with_equal']:.4f}")
        worst = sweep["worst_case"]
        print(f"  worst case over grid: cos={worst['cosine_with_equal']:.4f}")

    if hidden_specs:
        print("\n" + "=" * 66)
        print(" Bulk-direction diagnostic")
        print("=" * 66)
        diag = dominant_direction(hidden_specs, vectors, layer)
        out["bulk_direction"] = diag
        print(f"  boundary rows          {diag['boundary_rows_total']}")
        print(f"  ||bulk direction||     {diag['bulk_direction_norm']:.3f}")
        for name, c in diag["cosine_with_bulk_direction"].items():
            print(f"  cos({name}, bulk){c:>+9.4f}")
        print("\n  Cosine matrix with bulk direction projected out")
        _print_matrix(diag["cosine_matrix_bulk_removed"])
        print(
            f"\n  top-direction share    "
            f"{diag['span_bulk_removed']['top_direction_share']:.3f}"
        )
    else:
        print(
            "\n  (bulk-direction diagnostic skipped — pass --hidden to enable)"
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Norms, cosines, span structure and weight sensitivity "
        "for the SEAL calibration vectors."
    )
    parser.add_argument(
        "--vector",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Vector to include, repeatable. Defaults to the three domain "
        "vectors tracked in the repo.",
    )
    parser.add_argument(
        "--hidden",
        action="append",
        default=[],
        metavar="NAME=PATH[,PATH]",
        help="hidden.pt files for the bulk-direction diagnostic, repeatable. "
        "Normally the correct and incorrect files for one domain.",
    )
    parser.add_argument(
        "--layer", type=int, default=20, help="Hidden layer to read (default 20)."
    )
    parser.add_argument(
        "--weight-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHT_GRID),
        help="Weights to sweep in the sensitivity check (default 0.5 1 2).",
    )
    parser.add_argument(
        "--json", default=None, help="Also write the report to this JSON path."
    )
    args = parser.parse_args()

    vectors = load_vectors(args.vector or DEFAULT_VECTORS)
    out = report(vectors, args.hidden, args.layer, args.weight_grid)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
