"""Run the pre-GPU checklist on the R_iso control draws.

Every check here is CPU-only and runs in seconds, which is the point: a bad
control vector is much cheaper to catch now than after three MATH-500 arms have
been evaluated against it. The checklist is §9 of
``docs/random_vector_build_plan.html``.

What each check is actually guarding against:

``shape/dtype``
    The eval loads the tensor and adds it straight into the residual stream. A
    ``[1, 1536]`` tensor would broadcast rather than fail, silently steering
    something other than what was intended.

``norm``
    ``coef`` is fixed at −1.0, so the norm *is* the intervention strength. A raw
    ``randn(1536)`` has norm ~39.2; an unnormalized draw would be a ~0.7x
    intervention, and a unit draw a ~0.02x one. Both would read as "random
    steering does nothing" for entirely uninteresting reasons.

``cosine vs the SEAL vectors``
    Confirms the control is not accidentally a weak copy of the thing it is
    controlling for.

``pairwise cosine between draws``
    Three draws that happened to be similar would not be three independent
    samples, and the whole argument for building three collapses.

``reproducibility``
    Rebuilds each vector from its recorded seed and requires bit-identical
    output. This is what makes the meta a sufficient record: anyone can
    regenerate the exact tensor without the ``.pt`` file.

``meta agreement``
    Catches a ``.pt`` and a ``.meta.json`` that have drifted apart — e.g. a
    vector rebuilt with different flags but the old meta left in place.

Exits nonzero if any check fails, so it can gate a run script.

Usage::

    python scripts/verify_random_vectors.py
    python scripts/verify_random_vectors.py --dir results/control --prefix R_iso
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_general_vector import cosine  # noqa: E402
from build_random_vector import draw_iso, load_reference  # noqa: E402
from scripts.vector_geometry import DEFAULT_VECTORS, load_vectors  # noqa: E402

NORM_TOL = 1e-3

PASS = "  ok  "
FAIL = " FAIL "


class Checker:
    """Collects pass/fail lines so every check runs before anything exits."""

    def __init__(self) -> None:
        self.failures: List[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        print(f"[{PASS if ok else FAIL}] {label}" + (f"  — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)


def load_meta(vector_path: str) -> Dict[str, object]:
    """Read the sidecar meta for a vector, or {} if it is missing."""
    meta_path = os.path.splitext(vector_path)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as handle:
        return json.load(handle)


def rebuild_from_meta(meta: Dict[str, object], dim: int) -> torch.Tensor:
    """Regenerate a vector from its recorded seed, replaying any rejections.

    References are not reloaded — the rebuild does not re-run the cosine screen,
    it replays the generator. A rejected draw still advances the generator, so
    reaching the accepted draw means discarding ``attempts - 1`` draws first.
    Skipping that replay would silently rebuild the *first* draw whenever a
    rejection had occurred, and the check would fail for a reason that has
    nothing to do with reproducibility.

    Uses ``target_norm``, not ``vector_norm``: the latter is rounded for
    readability, and multiplying by a rounded scale reproduces a nearly
    identical vector rather than a bit-identical one.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(meta["seed"]))
    for _ in range(int(meta.get("attempts", 1)) - 1):
        draw_iso(dim, generator)
    return draw_iso(dim, generator) * float(meta["target_norm"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the R_iso control draws.")
    parser.add_argument("--dir", default="results/control", help="Vector directory.")
    parser.add_argument("--prefix", default="R_iso", help="Vector filename prefix.")
    parser.add_argument(
        "--max-cos",
        type=float,
        default=None,
        help="Override the threshold; defaults to each vector's recorded max_cos.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, f"{args.prefix}_seed*.pt")))
    if not paths:
        print(f"No vectors matching {args.prefix}_seed*.pt in {args.dir}")
        sys.exit(1)

    print("=" * 70)
    print(f" Verifying {len(paths)} control vector(s) in {args.dir}")
    print("=" * 70)

    checker = Checker()
    seal = load_vectors(DEFAULT_VECTORS)
    for name, path in [
        ("S_general", "results/general/S_general_math_apps_logic_phase1.pt"),
        ("S_combo", "results/general/S_combo.pt"),
    ]:
        if os.path.exists(path):
            seal[name] = load_reference(path)

    drawn: Dict[str, torch.Tensor] = {}

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n--- {name} ---")
        meta = load_meta(path)
        checker.check(bool(meta), f"{name}: meta present")

        vector = torch.load(path, map_location="cpu", weights_only=True)
        checker.check(
            vector.dim() == 1, f"{name}: is 1-D", f"shape {tuple(vector.shape)}"
        )
        checker.check(
            vector.dtype == torch.float32, f"{name}: float32", str(vector.dtype)
        )

        vector = vector.float().flatten()
        drawn[name] = vector

        if meta:
            target = float(meta["vector_norm"])
            got = float(vector.norm())
            checker.check(
                abs(got - target) <= NORM_TOL,
                f"{name}: norm matches meta",
                f"{got:.6f} vs {target:.6f}",
            )
            checker.check(
                int(meta["vector_dim"]) == int(vector.numel()),
                f"{name}: dim matches meta",
            )
            checker.check(
                float(meta["apply_coef"]) == -1.0 and int(meta["apply_layer"]) == 20,
                f"{name}: records layer 20 / coef -1.0",
            )

        max_cos = args.max_cos if args.max_cos is not None else float(
            meta.get("max_cos", 0.08)
        )
        worst_name, worst = "", 0.0
        for seal_name, seal_vec in seal.items():
            c = abs(cosine(vector, seal_vec))
            if c > worst:
                worst_name, worst = seal_name, c
        checker.check(
            worst <= max_cos,
            f"{name}: near-orthogonal to all {len(seal)} SEAL vectors",
            f"worst |cos| {worst:.4f} vs {worst_name}, threshold {max_cos}",
        )

        if meta:
            rebuilt = rebuild_from_meta(meta, int(vector.numel()))
            checker.check(
                torch.equal(rebuilt, vector),
                f"{name}: reproducible from seed {meta['seed']}",
                "bit-identical" if torch.equal(rebuilt, vector) else "DIFFERS",
            )

    print("\n--- pairwise between draws ---")
    names = sorted(drawn)
    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append((a, b, cosine(drawn[a], drawn[b])))
    for a, b, c in pairs:
        checker.check(abs(c) <= 0.08, f"{a} vs {b}", f"cos {c:+.4f}")
    if len(names) > 1:
        checker.check(
            all(not torch.equal(drawn[a], drawn[b]) for a, b, _ in pairs),
            "all draws are distinct tensors",
        )

    print("\n" + "=" * 70)
    if checker.failures:
        print(f" {len(checker.failures)} CHECK(S) FAILED — do not spend GPU time")
        for failure in checker.failures:
            print(f"   - {failure}")
        print("=" * 70)
        sys.exit(1)
    print(" All checks passed. Vectors are ready to evaluate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
