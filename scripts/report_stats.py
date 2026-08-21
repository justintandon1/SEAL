"""Per-problem statistics behind docs/steering_progress_report.html.

Every arm of a benchmark is evaluated on the same problems in the same order,
so the right test is the paired one. This script reproduces the analyses in the
report that are not stored in any ``metrics.json``:

1. **McNemar** (exact two-sided binomial on the discordant pairs) for every
   pair of arms. More sensitive than comparing two independent error bars,
   which is what a naive reading of an accuracy table would do.

2. **Token usage.** Average generated tokens per arm and the fraction of
   generations that hit ``--max_tokens``, using the model tokenizer.

3. **The budget-exhaustion split.** Partition the problems by whether the
   *baseline* hit the token cap, then score every arm on both halves, and
   decompose each arm's delta into the part contributed by each half. Across
   all six benchmarks the gain lives almost entirely in the cap-hit half.

4. **Uniform LogiQA grading.** LogiQA is scored from the raw generations with
   ``extract_choice`` for every arm, baseline included, so no row can depend on
   which grader happened to be current when that run was produced.

The tokenizer
    Token counts need ``transformers`` and the model's tokenizer files. When
    they are unavailable the script still runs: it falls back to an answer-marker
    proxy for the cap split (no ``\\boxed{}`` on MATH-500/GSM8K, no closing
    ``</think>`` elsewhere) and reports lengths in characters. On the MATH-500
    baseline the proxy flags 166 problems against the tokenizer's true 164, so
    conclusions do not change -- but prefer the real counts when you can.

Usage::

    python scripts/report_stats.py                      # everything
    python scripts/report_stats.py --benchmark math500
    python scripts/report_stats.py --results-root /path/to/results
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from math import comb, sqrt
from statistics import mean, median
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic_utils import extract_choice, options_from_prompt  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_7B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_TOKENS = 10000

# Arm -> path relative to the results root. Runs are filed under the vector that
# produced them (results_for_<source>_vectors/<benchmark>/), so one benchmark's
# arms are spread across several trees. See results/README.md.
BENCHMARKS: Dict[str, dict] = {
    "math500": {
        "label": "MATH-500 (Omni-MATH rule grader)",
        "eval_file": "math_eval.jsonl",
        "scorer": "math",
        "arms": {
            "baseline": "results_for_math_vectors/MATH500/baseline/base_remove_bos/rand42_500",
            "S_math": "results_for_math_vectors/MATH500/math_vector/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/rand42_500",
            "S_code": "results_for_code_vectors/MATH500/code_vector/vectors_apps_v_code/coef_-1.0_remove_bos/rand42_500",
            "S_logic": "results_for_logic_vectors/MATH500/logic_vector/baseline_3000_regraded_vector_logic_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/rand42_500",
            "S_general": "results_for_general_vectors/MATH500/s_general_phase1/results_general_S_general_math_apps_logic_phase1/coef_-1.0_remove_bos/rand42_500",
            "S_combo": "results_for_combo_vectors/MATH500/s_combo/results_general_S_combo/coef_-1.0_remove_bos/rand42_500",
            # Control A: norm-matched random directions, three draws. Registered
            # ahead of the runs — missing arms report as "not run" rather than
            # erroring, so these fill in as the pod finishes them. See
            # docs/random_vector_build_plan.html.
            "R_iso_seed1": "results_for_control_vectors/MATH500/R_iso_seed1/results_control_R_iso_seed1/coef_-1.0_remove_bos/rand42_500",
            "R_iso_seed2": "results_for_control_vectors/MATH500/R_iso_seed2/results_control_R_iso_seed2/coef_-1.0_remove_bos/rand42_500",
            "R_iso_seed3": "results_for_control_vectors/MATH500/R_iso_seed3/results_control_R_iso_seed3/coef_-1.0_remove_bos/rand42_500",
        },
    },
    # The 7B arms live in their own tree and need their own tokenizer -- the two
    # models do not share a vocabulary, so counting 7B generations with the 1.5B
    # tokenizer would be wrong rather than merely imprecise. R_iso_7b is
    # registered ahead of its runs, same as the 1.5B control arms above.
    "math500_7b": {
        "label": "MATH-500 on DeepSeek-R1-Distill-Qwen-7B",
        "eval_file": "math_eval.jsonl",
        "scorer": "math",
        "model": MODEL_7B,
        "arms": {
            "baseline": "results_for_7b_math_vectors/MATH500/baseline/base_remove_bos/rand42_500",
            "S_math_7b": "results_for_7b_math_vectors/MATH500/math_vector_7b/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/rand42_500",
            "R_iso_7b_seed1": "results_for_control_vectors/MATH500_7b/R_iso_7b_seed1/results_control_R_iso_7b_seed1/coef_-1.0_remove_bos/rand42_500",
            "R_iso_7b_seed2": "results_for_control_vectors/MATH500_7b/R_iso_7b_seed2/results_control_R_iso_7b_seed2/coef_-1.0_remove_bos/rand42_500",
            "R_iso_7b_seed3": "results_for_control_vectors/MATH500_7b/R_iso_7b_seed3/results_control_R_iso_7b_seed3/coef_-1.0_remove_bos/rand42_500",
        },
    },
    "logiqa": {
        "label": "LogiQA clean-500 (reasoning grader applied to every arm)",
        "eval_file": "predictions.jsonl",
        "scorer": "logiqa",
        "arms": {
            "baseline": "results_for_logic_vectors/LogiQA/baseline/base_remove_bos/rand42_500_clean",
            "S_math": "results_for_math_vectors/LogiQA/math_vector/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/eval_rand42_500_clean",
            "S_code": "results_for_code_vectors/LogiQA/code_vector/vectors_apps_v_code/coef_-1.0_remove_bos/eval_rand42_500_clean",
            "S_logic": "results_for_logic_vectors/LogiQA/logic_vector/baseline_3000_regraded_vector_logic_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/eval_rand42_500_clean",
            "S_general": "results_for_general_vectors/LogiQA/s_general_phase1/results_general_S_general_math_apps_logic_phase1/coef_-1.0_remove_bos/eval_rand42_500_clean",
            "S_combo": "results_for_combo_vectors/LogiQA/s_combo/results_general_S_combo/coef_-1.0_remove_bos/eval_rand42_500_clean",
            # Control A on LogiQA. Registered ahead of the runs, like the
            # MATH-500 arms above -- missing arms report as "not run" rather
            # than erroring. Note the LogiQA effect is 13-20 problems out of
            # 500, against 75 on MATH-500, so read a null here accordingly.
            "R_iso_seed1": "results_for_control_vectors/LogiQA/R_iso_seed1/results_control_R_iso_seed1/coef_-1.0_remove_bos/eval_rand42_500_clean",
            "R_iso_seed2": "results_for_control_vectors/LogiQA/R_iso_seed2/results_control_R_iso_seed2/coef_-1.0_remove_bos/eval_rand42_500_clean",
            "R_iso_seed3": "results_for_control_vectors/LogiQA/R_iso_seed3/results_control_R_iso_seed3/coef_-1.0_remove_bos/eval_rand42_500_clean",
        },
    },
    "apps": {
        "label": "APPS test rand42-500 (pass@1, tests executed)",
        "eval_file": "code_eval.jsonl",
        "scorer": "code",
        "arms": {
            "baseline": "results_for_math_vectors/APPS/baseline/base_remove_bos/test_rand42_500",
            "S_math": "results_for_math_vectors/APPS/math_vector/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/test_rand42_500",
            "S_code": "results_for_code_vectors/APPS/code_vector/vectors_apps_v_code/coef_-1.0_remove_bos/test_rand42_500",
            "S_logic": "results_for_logic_vectors/APPS/logic_vector/baseline_3000_regraded_vector_logic_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/test_rand42_500",
            "S_general": "results_for_general_vectors/APPS/s_general_phase1/results_general_S_general_math_apps_logic_phase1/coef_-1.0_remove_bos/test_rand42_500",
            "S_combo": "results_for_combo_vectors/APPS/s_combo/results_general_S_combo/coef_-1.0_remove_bos/test_rand42_500",
            # Control A on APPS, phase 1 of the two-phase plan (2026-08-17).
            # Only seed 1 is registered here: phase 2 brings seeds 2 and 3 back
            # on MATH-500 first, and extends outward only if budget allows.
            # Registered ahead of the run -- a missing arm reports as "not run"
            # rather than erroring. The path shape is confirmed by the n=5 smoke
            # under results_for_control_vectors/smoke_apps/, which differs only
            # in the trailing selection dir. See docs/random_vector_build_plan.html.
            "R_iso_seed1": "results_for_control_vectors/APPS/R_iso_seed1/results_control_R_iso_seed1/coef_-1.0_remove_bos/test_rand42_500",
        },
    },
    "math500_7b": {
        "label": "MATH-500 7B (DeepSeek-R1-Distill-Qwen-7B, Omni-MATH rule grader)",
        "eval_file": "math_eval.jsonl",
        "scorer": "math",
        "arms": {
            # The baseline and S_math arms are the original-environment runs
            # whose per-problem artifacts live in results_for_7b_math_vectors.
            # R_iso_7b_seed1 was generated at batch 4 on a different box; the
            # batch cross-check (build plan §14) showed batching is exact
            # same-box, but cross-environment token streams diverge, so
            # comparisons against the original arms carry that caveat.
            "baseline": "results_for_7b_math_vectors/MATH500/baseline/base_remove_bos/rand42_500",
            "S_math": "results_for_7b_math_vectors/MATH500/math_vector_7b/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/rand42_500",
            "R_iso_7b_seed1": "results_for_control_vectors_7b/MATH500/R_iso_7b_seed1/results_control_R_iso_7b_seed1/coef_-1.0_remove_bos/rand42_500",
        },
    },
    "logiqa_7b": {
        "label": "LogiQA clean-500 7B (reasoning grader applied to every arm)",
        "eval_file": "predictions.jsonl",
        "scorer": "logiqa",
        "model": MODEL_7B,
        "arms": {
            # Pairing is pinned by eval_rand42_500_clean (PR #69). Do not mix
            # with logiqa_7b_legacy (dirty eval_rand42_500.json).
            "baseline": "results_for_7b_logic_vectors/LogiQA/baseline/base_remove_bos/eval_rand42_500_clean",
            "S_logic": "results_for_7b_logic_vectors/LogiQA/logic_vector_7b/workspace_vectors_logiqa2_v_logic/coef_-1.0_remove_bos/eval_rand42_500_clean",
            "R_iso_7b_seed1": "results_for_control_vectors_7b/LogiQA/R_iso_7b_seed1/results_control_R_iso_7b_seed1/coef_-1.0_remove_bos/eval_rand42_500_clean",
        },
    },
    # Justin 7B S_logic MATH/APPS transfer. Own baselines (MATH 0.850, APPS
    # 0.284) — not Akhilesh's math500_7b 0.844 MATH job.
    "math500_7b_s_logic": {
        "label": "MATH-500 7B, S_logic cell (Omni-MATH rule grader)",
        "eval_file": "math_eval.jsonl",
        "scorer": "math",
        "model": MODEL_7B,
        "arms": {
            "baseline": "results_for_7b_logic_vectors/MATH500/baseline/base_remove_bos/rand42_500",
            "S_logic": "results_for_7b_logic_vectors/MATH500/logic_vector_7b/workspace_vectors_logiqa2_v_logic/coef_-1.0_remove_bos/rand42_500",
        },
    },
    "apps_7b_s_logic": {
        "label": "APPS test rand42-500 7B, S_logic cell (pass@1)",
        "eval_file": "code_eval.jsonl",
        "scorer": "code",
        "model": MODEL_7B,
        "arms": {
            "baseline": "results_for_7b_logic_vectors/APPS/baseline/base_remove_bos/test_rand42_500",
            "S_logic": "results_for_7b_logic_vectors/APPS/logic_vector_7b/workspace_vectors_logiqa2_v_logic/coef_-1.0_remove_bos/test_rand42_500",
        },
    },
    # 7B S_math → LogiQA transfer, run before the clean selection existed.
    # Different 500 than logiqa_7b (16 train overlaps in eval_rand42_500.json).
    # Keep it a separate benchmark so McNemar never pairs mismatched problems.
    "logiqa_7b_legacy": {
        "label": "LogiQA n=500 seed 42 7B (legacy eval_rand42_500.json, 16 train overlaps)",
        "eval_file": "predictions.jsonl",
        "scorer": "logiqa",
        "model": MODEL_7B,
        "arms": {
            "baseline": "results_for_7b_math_vectors/LogiQA/baseline/base_remove_bos/rand42_500",
            "S_math_7b": "results_for_7b_math_vectors/LogiQA/math_vector_7b/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/rand42_500",
        },
    },
    "mbpp": {
        "label": "MBPP test-500 (pass@1, tests executed)",
        "eval_file": "code_eval.jsonl",
        "scorer": "code",
        "arms": {
            "baseline": "results_for_math_vectors/MBPP/DeepSeek-R1-Distill-Qwen-1.5B/transfer_baseline",
            "S_math": "results_for_math_vectors/MBPP/DeepSeek-R1-Distill-Qwen-1.5B/transfer_steered/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos",
            "S_code": "results_for_code_vectors/MBPP/code_vector/vectors_apps_v_code/coef_-1.0_remove_bos/test_500",
        },
    },
    "gsm8k": {
        "label": "GSM8K full test (n=1319)",
        "eval_file": "math_eval.jsonl",
        "scorer": "math",
        "arms": {
            "baseline": "results_for_math_vectors/GSM/DeepSeek-R1-Distill-Qwen-1.5B/paper_baseline_10000",
            "S_math": "results_for_math_vectors/GSM/DeepSeek-R1-Distill-Qwen-1.5B/paper_steer_10000/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos",
            "S_code": "results_for_code_vectors/GSM/code_vector/vectors_apps_v_code/coef_-1.0_remove_bos/test_1319",
        },
    },
    "mmlu": {
        "label": "MMLU philosophy (n=311)",
        "eval_file": "math_eval.jsonl",
        "scorer": "math",
        "arms": {
            "baseline": "results_for_math_vectors/MMLU/DeepSeek-R1-Distill-Qwen-1.5B/transfer_baseline",
            "S_math": "results_for_math_vectors/MMLU/DeepSeek-R1-Distill-Qwen-1.5B/transfer_steered/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos",
        },
    },
}


_TOKENIZERS: Dict[str, Optional[object]] = {}


def get_tokenizer(model: str = MODEL):
    """Return ``model``'s tokenizer, or None if transformers/files are unavailable.

    Cached per model: benchmarks are reported in a loop and the 1.5B tokenizer
    would otherwise be reloaded once per benchmark.
    """
    if model in _TOKENIZERS:
        return _TOKENIZERS[model]
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        print(f"  [warn] tokenizer for {model} unavailable ({type(exc).__name__}); "
              f"lengths in characters, cap split uses the answer-marker proxy")
        tok = None
    _TOKENIZERS[model] = tok
    return tok


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial(b + c, 0.5) on the discordants.

    Args:
        b: Problems the first arm got right and the second got wrong.
        c: Problems the first arm got wrong and the second got right.

    Returns:
        The p-value, or 1.0 when the arms never disagree.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def score_math(row: dict) -> bool:
    """Correctness from ``mv_eval``, already written by the rule grader."""
    return bool(row["mv_eval"])


def score_code(row: dict) -> bool:
    """Correctness from ``all_eval``, written by executing the unit tests."""
    return bool(row["all_eval"][0])


def score_logiqa(row: dict) -> bool:
    """Score from the raw generation with the current reasoning extractor.

    Scoring here (rather than trusting a stored ``mv_eval``) is deliberate: it
    applies one grading rule to every arm at analysis time, so the rows cannot
    drift apart because runs were graded at different times. The rule returns
    no answer -- not a guess -- when the generation never closed ``</think>``.
    """
    pred = extract_choice(
        row["model_generation"][0], options=options_from_prompt(row.get("prompt", ""))
    )
    return pred is not None and pred == int(row["answer"])


SCORERS: Dict[str, Callable[[dict], bool]] = {
    "math": score_math,
    "code": score_code,
    "logiqa": score_logiqa,
}
# Marker whose absence means the model never reached an answer. Used only when
# no tokenizer is available.
PROXY_MARKER = {"math": "\\boxed", "code": "</think>", "logiqa": "</think>"}


def report(root: str, key: str, spec: dict, tokenizer) -> None:
    print("=" * 78)
    print(f" {spec['label']}")
    print("=" * 78)

    correct: Dict[str, List[bool]] = {}
    lengths: Dict[str, List[int]] = {}
    at_cap: Dict[str, List[bool]] = {}
    identity: Dict[str, List] = {}
    scorer = SCORERS[spec["scorer"]]
    marker = PROXY_MARKER[spec["scorer"]]

    for name, rel in spec["arms"].items():
        path = os.path.join(root, rel, spec["eval_file"])
        if not os.path.exists(path):
            print(f"  [skip] {name}: not run (missing {path})")
            continue
        rows = [json.loads(line) for line in open(path)]
        gens = [r["model_generation"][0] for r in rows]
        correct[name] = [scorer(r) for r in rows]
        identity[name] = [r.get("problem", r.get("problem_id")) for r in rows]
        if tokenizer is not None:
            lengths[name] = [
                len(tokenizer.encode(g, add_special_tokens=False)) for g in gens
            ]
            at_cap[name] = [n >= MAX_TOKENS - 10 for n in lengths[name]]
        else:
            lengths[name] = [len(g) for g in gens]
            at_cap[name] = [marker not in g for g in gens]

    if not correct:
        print("  no runs found\n")
        return

    reference = identity.get("baseline")
    if reference is not None:
        mismatched = [n for n, p in identity.items() if p != reference]
        if mismatched:
            raise ValueError(
                "arms are not scored on the same problems in the same order, so "
                f"the paired tests would be invalid: {mismatched}"
            )

    names = list(correct)
    n = len(correct[names[0]])
    unit = "tok" if tokenizer is not None else "chars"
    base_len = mean(lengths["baseline"]) if "baseline" in lengths else None

    print(f"\n  accuracy and token usage (n={n})\n")
    print(f"  {'arm':<11}{'acc':>8}{'95% CI':>9}{'avg ' + unit:>10}"
          f"{'median':>9}{'Δ len':>9}{'at cap':>9}")
    for name in names:
        acc = mean(correct[name])
        ci = 1.96 * sqrt(acc * (1 - acc) / n)
        avg = mean(lengths[name])
        delta = 100 * (avg - base_len) / base_len if base_len else 0.0
        print(
            f"  {name:<11}{acc * 100:7.1f}%{ci * 100:9.1f}{avg:10.0f}"
            f"{median(lengths[name]):9.0f}{delta:+8.1f}%"
            f"{mean(at_cap[name]) * 100:8.1f}%"
        )

    print("\n  McNemar, exact two-sided\n")
    print(f"  {'pair':<26}{'A✓B✗':>6}{'A✗B✓':>6}{'Δ pts':>8}{'p':>12}")
    for a, b in itertools.combinations(names, 2):
        A, B = correct[a], correct[b]
        n01 = sum(1 for x, y in zip(A, B) if x and not y)
        n10 = sum(1 for x, y in zip(A, B) if not x and y)
        print(
            f"  {a + ' vs ' + b:<26}{n01:6d}{n10:6d}{(mean(B) - mean(A)) * 100:+8.1f}"
            f"{mcnemar_exact(n01, n10):12.4g}"
        )

    if "baseline" in at_cap:
        split = at_cap["baseline"]
        n_cap, n_fin = sum(split), len(split) - sum(split)
        print(
            f"\n  split by whether the BASELINE hit the {MAX_TOKENS}-token cap"
            f"  (cap-hit={n_cap}, finished={n_fin})\n"
        )
        print(f"  {'arm':<11}{'cap-hit':>10}{'finished':>10}{'overall':>10}"
              f"{'from cap':>10}{'from fin':>10}")
        base_cap = mean([c for c, t in zip(correct["baseline"], split) if t])
        base_fin = mean([c for c, t in zip(correct["baseline"], split) if not t])
        for name in names:
            cap = mean([c for c, t in zip(correct[name], split) if t])
            fin = mean([c for c, t in zip(correct[name], split) if not t])
            # Each half's contribution to this arm's overall delta, in points.
            g_cap = (cap - base_cap) * n_cap / len(split) * 100
            g_fin = (fin - base_fin) * n_fin / len(split) * 100
            print(
                f"  {name:<11}{cap * 100:9.1f}%{fin * 100:9.1f}%"
                f"{mean(correct[name]) * 100:9.1f}%{g_cap:+9.1f}{g_fin:+9.1f}"
            )

    steered = [name for name in names if name != "baseline"]
    if len(steered) > 1:
        both = sum(1 for i in range(n) if all(correct[k][i] for k in steered))
        neither = sum(1 for i in range(n) if not any(correct[k][i] for k in steered))
        print(
            f"\n  all {len(steered)} steered arms agree on {both + neither}/{n} "
            f"problems ({(both + neither) / n * 100:.1f}%): "
            f"{both} all-correct, {neither} all-wrong"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory holding the results_for_* trees (default: results).",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS) + ["all"],
        default="all",
        help="Which benchmark to report (default: all).",
    )
    args = parser.parse_args()

    keys = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    for key in keys:
        spec = BENCHMARKS[key]
        report(args.results_root, key, spec, get_tokenizer(spec.get("model", MODEL)))


if __name__ == "__main__":
    main()
