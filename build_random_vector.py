"""Build ``R_iso``, the norm-matched random-direction control (Control A).

The steering vectors gain +12–15 points on MATH-500, but §11 of the progress
report shows the mechanism is *compression*: the budget-limited pile (baseline
hit the token cap) supplies at least 100% of the net gain on every benchmark,
while the finished pile contributes between −4.2 and +0.6. So the obvious
alternative explanation is that any perturbation of this magnitude, at this
layer, shortens chains of thought — and the reasoning story is incidental.

This script builds the control that tests it::

    R_iso = target_norm * g / ||g||,   g ~ N(0, I_d)

drawn from a pinned seed, then evaluated exactly like the SEAL arms: layer 20,
``coef = -1.0``, ``--remove_bos``. Everything is matched except the direction.

Why the norm is matched, not the coefficient
    ``eval_*_steering.py`` applies ``steer_coef * steer_vec`` at a fixed
    ``coef = -1.0``, so the vector's norm *is* the intervention strength. A
    control at the wrong magnitude tests nothing: too small is inert by
    construction, too large degrades the model for reasons unrelated to
    direction. The default target is 57.06 — the mean of the three domain-vector
    norms, the same ``mean-source`` figure ``S_combo`` uses. The five real
    vectors span 55.29–59.67, so landing mid-band means no outcome can be
    attributed to strength.

Why several draws
    "Random vector" is a category, not an object. One draw is n=1: flat tells
    you nothing about random directions in general, and hot cannot be
    distinguished from one unlucky draw. Build seeds 1/2/3 and read the pattern.
    Three draws also settle the sign question — a contrast vector has a real
    orientation, a random one does not, and ``-R_seed1`` is drawn from the same
    distribution as ``R_seed2``, so three draws at ``coef = -1.0`` already
    sample both half-spaces.

The redraw rule is pre-registered, not improvised
    In d = 1536 the cosine between a random direction and any fixed vector
    concentrates at 1/sqrt(d) = 0.0255 (measured over 20k draws against S_math:
    mean +0.0001, sd 0.0251). ``--max-cos 0.08`` is ~3.1 sigma, so it fires on
    about 0.21% of draws and will almost certainly never trigger. That is the
    point: a discard rule invented *after* seeing a hot result is the same rule
    with none of the credibility. Rejections are recorded in the meta.

Isotropic is the *easy* control
    The residual stream is anisotropic, so a Gaussian draw is nearly orthogonal
    to the region the model actually computes in — ``R_iso`` may come back flat
    for reasons having nothing to do with reasoning. The harder control is
    ``R_cov``, drawn from a Gaussian matched to the covariance of layer-20
    boundary activations; it needs ``hidden.pt`` and is staged as the follow-on.
    ``--mode`` exists so that lands here rather than in a second script.

Sign convention — deliberately absent
    ``build_combo_vector.check_sign_convention`` aborts on negative cosines,
    which is correct when summing contrast vectors: opposed inputs cancel. It
    must NOT be reused here. A random vector sits at cos ~ 0 and is negative
    about half the time, so that guard would reject half of all valid draws and
    silently bias the control into the SEAL vectors' half-space. This script
    checks |cos| against a symmetric threshold instead.

Examples::

    # The three primary draws. Each build also checks against the ones before
    # it, so the seeds are verified mutually near-orthogonal as they are made.
    python build_random_vector.py --seed 1 \\
        --like results/general/S_combo.pt \\
        --out results/control/R_iso_seed1.pt

    python build_random_vector.py --seed 2 \\
        --like results/general/S_combo.pt \\
        --check-against results/control/R_iso_seed1.pt \\
        --out results/control/R_iso_seed2.pt

    # A stronger perturbation, for the deferred magnitude sweep (build plan §11).
    python build_random_vector.py --seed 1 --like results/general/S_combo.pt \\
        --norm 114.12 --out results/control/R_iso_seed1_norm2x.pt

Full rationale: docs/random_vector_build_plan.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_general_vector import cosine  # noqa: E402
from scripts.vector_geometry import DEFAULT_VECTORS, load_vectors  # noqa: E402

# Matches the SEAL arms. Recorded in the meta so a run can be reproduced from
# the vector alone, and so a mismatched evaluation is auditable after the fact.
APPLY_COEF = -1.0
APPLY_LAYER = 20

# ~3.1 sigma on the d=1536 cosine null. See the module docstring.
DEFAULT_MAX_COS = 0.08

# A draw this far out means a bug, not luck; refusing to loop forever surfaces
# it instead of silently burning attempts.
MAX_ATTEMPTS = 20


def parse_reference(spec: str) -> Tuple[str, str]:
    """Parse ``--check-against`` as ``name=path`` or a bare ``path``.

    Args:
        spec: Either ``name=path`` or ``path``. A bare path is named after its
            file stem, so ``vectors/apps_v_code.pt`` becomes ``apps_v_code``.

    Returns:
        ``(name, path)``.

    Raises:
        argparse.ArgumentTypeError: If the spec is empty or names nothing.
    """
    spec = spec.strip()
    if not spec:
        raise argparse.ArgumentTypeError("--check-against got an empty value")
    if "=" in spec:
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise argparse.ArgumentTypeError(
                f"--check-against expected name=path, got: {spec!r}"
            )
        return name, path
    return os.path.splitext(os.path.basename(spec))[0], spec


def load_reference(path: str) -> torch.Tensor:
    """Load a comparison vector as a flat float tensor."""
    return torch.load(path, map_location="cpu", weights_only=True).float().flatten()


def resolve_target_norm(norm_mode: str, like: torch.Tensor | None) -> Tuple[float, str]:
    """Resolve ``--norm`` into a concrete magnitude.

    Args:
        norm_mode: ``like`` copies the norm of ``--like``; ``mean-source``
            averages the three tracked domain-vector norms (the figure
            ``S_combo`` is scaled to); or an explicit number.
        like: The tensor given by ``--like``, if any.

    Returns:
        ``(target_norm, human_readable_source)``.

    Raises:
        ValueError: If the mode needs ``--like`` and it was not given, or if the
            mode is not recognized.
    """
    if norm_mode == "like":
        if like is None:
            raise ValueError(
                "--norm like needs --like PATH. Pass a vector to copy the "
                "magnitude from, or give --norm a number or 'mean-source'."
            )
        return float(like.norm()), "--like"

    if norm_mode == "mean-source":
        sources = load_vectors(DEFAULT_VECTORS)
        norms = {name: float(v.norm()) for name, v in sources.items()}
        return sum(norms.values()) / len(norms), (
            "mean of " + ", ".join(f"{k}={v:.3f}" for k, v in norms.items())
        )

    try:
        return float(norm_mode), "explicit"
    except ValueError as exc:
        raise ValueError(
            f"--norm expected 'like', 'mean-source' or a number, got {norm_mode!r}"
        ) from exc


def draw_iso(dim: int, generator: torch.Generator) -> torch.Tensor:
    """Draw one isotropic unit direction.

    A raw ``randn(dim)`` has norm ~sqrt(dim) — 39.2 at dim=1536, neither 1 nor
    the target — so normalizing here keeps the scaling in one place and makes a
    magnitude error impossible to introduce downstream.
    """
    g = torch.randn(dim, generator=generator, dtype=torch.float32)
    return g / g.norm()


def build_vector(
    dim: int,
    seed: int,
    mode: str,
    target_norm: float,
    references: Dict[str, torch.Tensor],
    max_cos: float,
) -> Tuple[torch.Tensor, List[Dict[str, object]], int]:
    """Draw until a direction clears the cosine threshold, then scale it.

    Rejected draws are not discarded quietly: the generator advances, so the
    accepted vector is "the first acceptable draw from the generator seeded
    ``seed``" — still fully reproducible from the seed, with every rejection
    recorded for the write-up.

    Args:
        dim: Hidden dimension.
        seed: Pins the generator. CPU generator, so the draw does not depend on
            GPU availability or on what else consumed random numbers first.
        mode: ``iso`` today; ``cov`` is the staged follow-on.
        target_norm: Final magnitude.
        references: Vectors the draw must be near-orthogonal to.
        max_cos: Reject a draw whose |cos| against any reference exceeds this.

    Returns:
        ``(vector, rejections, attempts)``.

    Raises:
        NotImplementedError: For ``--mode cov``, which needs activations.
        RuntimeError: If no draw is accepted within ``MAX_ATTEMPTS``.
    """
    if mode == "cov":
        raise NotImplementedError(
            "--mode cov is the staged follow-on control and needs layer-20 "
            "boundary activations (hidden.pt), which are not in the repo. See "
            "docs/random_vector_build_plan.html §5."
        )
    if mode != "iso":
        raise ValueError(f"unknown --mode {mode!r}; expected 'iso' or 'cov'")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    rejections: List[Dict[str, object]] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        direction = draw_iso(dim, generator)
        cosines = {name: cosine(direction, ref) for name, ref in references.items()}
        hot = {n: c for n, c in cosines.items() if abs(c) > max_cos}
        if not hot:
            vector = direction * target_norm
            # A silent factor-of-39 error would look exactly like "random
            # steering destroys the model", so fail loudly instead.
            if abs(float(vector.norm()) - target_norm) > 1e-3:
                raise RuntimeError(
                    f"norm check failed: got {float(vector.norm()):.6f}, "
                    f"expected {target_norm:.6f}"
                )
            return vector, rejections, attempt
        rejections.append(
            {"attempt": attempt, "cosines": {n: round(c, 6) for n, c in hot.items()}}
        )

    raise RuntimeError(
        f"no draw cleared |cos| <= {max_cos} in {MAX_ATTEMPTS} attempts. At "
        f"d={dim} this should happen roughly never — check that --check-against "
        f"points at real vectors and not at copies of the draw."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build R_iso, the norm-matched random-direction control."
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Pins the draw; recorded in the meta."
    )
    parser.add_argument(
        "--mode",
        default="iso",
        choices=["iso", "cov"],
        help="'iso' (default) draws N(0, I). 'cov' is the staged follow-on.",
    )
    parser.add_argument(
        "--like",
        default=None,
        metavar="PATH",
        help="Copy dimension and (by default) norm from this vector, so 1536 "
        "and 57.06 are never typed by hand.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Hidden dimension, if --like is not given.",
    )
    parser.add_argument(
        "--norm",
        default="like",
        help="Output magnitude: 'like' (default, copies --like), 'mean-source' "
        "(mean of the three domain vectors), or a number for the sweep.",
    )
    parser.add_argument(
        "--check-against",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="Vector the draw must be near-orthogonal to, repeatable. Pass the "
        "SEAL vectors and any earlier seeds.",
    )
    parser.add_argument(
        "--max-cos",
        type=float,
        default=DEFAULT_MAX_COS,
        help=f"Pre-registered redraw threshold on |cos| (default {DEFAULT_MAX_COS}).",
    )
    parser.add_argument("--out", required=True, help="Output .pt path.")
    args = parser.parse_args()

    like = load_reference(args.like) if args.like else None
    if like is None and args.dim is None:
        parser.error("give --like PATH or --dim N so the dimension is known")
    dim = args.dim if args.dim is not None else int(like.numel())
    if like is not None and args.dim is not None and int(like.numel()) != args.dim:
        parser.error(
            f"--dim {args.dim} disagrees with --like ({int(like.numel())}). "
            f"Drop one of them."
        )

    target_norm, norm_source = resolve_target_norm(args.norm, like)

    references: Dict[str, torch.Tensor] = {}
    for spec in args.check_against:
        name, path = parse_reference(spec)
        references[name] = load_reference(path)
    for name, ref in references.items():
        if int(ref.numel()) != dim:
            parser.error(
                f"--check-against {name} has dim {int(ref.numel())}, expected {dim}"
            )

    vector, rejections, attempts = build_vector(
        dim, args.seed, args.mode, target_norm, references, args.max_cos
    )
    cosines = {name: cosine(vector, ref) for name, ref in references.items()}

    print("=" * 66)
    print(f" {os.path.splitext(os.path.basename(args.out))[0]}")
    print("=" * 66)
    print(f"  mode          {args.mode}")
    print(f"  seed          {args.seed}")
    print(f"  dim           {dim}")
    print(f"  target norm   {target_norm:.4f}  ({norm_source})")
    print(f"  actual norm   {float(vector.norm()):.4f}")
    print(f"  attempts      {attempts}  ({len(rejections)} rejected)")

    if references:
        print(f"\n  cosine with references (reject if |cos| > {args.max_cos})")
        for name, c in cosines.items():
            print(f"    {name:<28} {c:+.4f}")
    else:
        print("\n  no --check-against given; cosine screen skipped")

    for entry in rejections:
        print(f"  ! attempt {entry['attempt']} rejected: {entry['cosines']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(vector, args.out)

    meta = {
        "name": os.path.splitext(os.path.basename(args.out))[0],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "recipe": "R = target_norm * g / ||g||, g ~ N(0, I_d)",
        "role": "Control A - norm-matched random direction",
        "mode": args.mode,
        "seed": args.seed,
        "attempts": attempts,
        "rejected_draws": rejections,
        "max_cos": args.max_cos,
        "torch_version": torch.__version__,
        "generator_device": "cpu",
        "apply_coef": APPLY_COEF,
        "apply_layer": APPLY_LAYER,
        "norm_mode": args.norm,
        "norm_source": norm_source,
        # Full precision, deliberately unrounded: this is the number a rebuild
        # multiplies by, so a rounded copy reproduces a *nearly* identical
        # vector rather than a bit-identical one. vector_norm below is the
        # rounded, human-readable measurement of the result.
        "target_norm": target_norm,
        "vector_dim": dim,
        "vector_norm": round(float(vector.norm()), 6),
        "cosine_with_references": {k: round(v, 6) for k, v in cosines.items()},
        "references": {
            name: path
            for name, path in (parse_reference(s) for s in args.check_against)
        },
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    print("\n" + "=" * 66)
    print(f"  vector : {args.out}")
    print(f"  meta   : {meta_path}")
    print(f"  apply  : layer {APPLY_LAYER}, coef {APPLY_COEF}, --remove_bos")
    print("=" * 66)


if __name__ == "__main__":
    main()
