"""Re-grade the LogiQA extraction set with the fixed grader, keeping every trace.

The original build labelled correct/incorrect with extract_choice_legacy, whose
fallback assigns a letter to generations that never closed </think>. That is a
GRADING bug: it fabricates an answer where none was stated.

Fixing the grader does NOT mean removing those traces. A generation that never
closed </think> is a failed attempt -> INCORRECT, and it stays in the pool:

  * Alignment. v_math and v_code exclude no traces; every attempt is either
    correct or incorrect. Dropping traces would make the logic pool a different
    population from the other domains'.
  * Signal. vector_generation pools boundaries from BOTH files into one
    check/switch/other contrast -- the correct/incorrect label is only a sampling
    control and never enters S = mean(check u switch) - mean(other). A
    non-terminating trace is dense, sustained reflection, i.e. exactly the
    behaviour the vector is meant to capture.

Row order is preserved, so hidden_analysis.py's `--sample N --start 0` keeps its
first-in-first-fill (greedy) selection.

    python scripts/relabel_logiqa_build.py --build_dir <dir> --out_dir <dir>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic_utils import extract_choice, options_from_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(os.path.join(args.build_dir, "data.jsonl"))]
    ev = [json.loads(l) for l in open(os.path.join(args.build_dir, "math_eval.jsonl"))]
    if len(data) != len(ev):
        raise SystemExit(f"data.jsonl has {len(data)} rows, math_eval.jsonl has {len(ev)}")

    out_data, out_ev = [], []
    n_unfinished = n_flipped = 0
    for d, e in zip(data, ev):
        assert d["problem"] == e["problem"], "data/math_eval misaligned"
        gen = e["model_generation"][0]
        gt = int(e["answer"])
        pred = extract_choice(gen, options=options_from_prompt(e.get("prompt", "")))
        if pred is None:
            n_unfinished += 1          # kept, and scored incorrect below
        ok = pred is not None and pred == gt
        if ok != bool(e["all_eval"][0]):
            n_flipped += 1
        e = dict(e)
        e["all_eval"] = [ok]
        e["all_pred"] = [pred]
        e["unfinished"] = pred is None
        out_data.append(d)
        out_ev.append(e)

    n_correct = sum(1 for e in out_ev if e["all_eval"][0])
    n_incorrect = len(out_ev) - n_correct
    n_unf_in_incorrect = sum(1 for e in out_ev if e["unfinished"])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "data.jsonl"), "w") as f:
        for d in out_data:
            f.write(json.dumps(d) + "\n")
    with open(os.path.join(args.out_dir, "math_eval.jsonl"), "w") as f:
        for e in out_ev:
            f.write(json.dumps(e) + "\n")

    print(f"input traces            : {len(ev)}")
    print(f"kept                    : {len(out_ev)}  (none dropped)")
    print(f"  correct               : {n_correct}")
    print(f"  incorrect             : {n_incorrect}")
    print(f"    of which unfinished : {n_unf_in_incorrect}")
    print(f"    of which answered   : {n_incorrect - n_unf_in_incorrect}")
    print(f"labels flipped vs legacy: {n_flipped}")
    print(f"\nfirst-500 pools (what --sample 500 --start 0 will take, file order):")
    seen_c = seen_i = 0
    unf_in_first500_incorrect = 0
    for e in out_ev:
        if e["all_eval"][0]:
            if seen_c < 500:
                seen_c += 1
        else:
            if seen_i < 500:
                seen_i += 1
                if e["unfinished"]:
                    unf_in_first500_incorrect += 1
    print(f"  correct pool fills 500 : {'YES' if seen_c >= 500 else f'NO ({seen_c})'}")
    print(f"  incorrect pool fills 500: {'YES' if seen_i >= 500 else f'NO ({seen_i})'}")
    print(f"  unfinished inside the first-500 incorrect: {unf_in_first500_incorrect}")
    print(f"wrote -> {args.out_dir}")


if __name__ == "__main__":
    main()
