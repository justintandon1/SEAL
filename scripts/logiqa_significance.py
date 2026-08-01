"""Paired McNemar tests across the four LogiQA arms (same 500 problems, same grader)."""
import glob
import json
import math
import sys

REPO = "/Users/andwwy/Desktop/Algoverse/SEAL dup/SEAL"
sys.path.insert(0, REPO)
from logic_utils import extract_choice, options_from_prompt

ARMS = {
    "baseline": f"{REPO}/results/results_for_logic_vectors/LogiQA/baseline/base_remove_bos/rand42_500_clean",
    "v_logic": glob.glob(f"{REPO}/results/results_for_logic_vectors/LogiQA/logic_vector/*/coef_-1.0_remove_bos/eval_rand42_500_clean")[0],
    "v_code": glob.glob(f"{REPO}/results/results_for_code_vectors/LogiQA/code_vector/*/*/*")[0],
    "v_math": glob.glob(f"{REPO}/results/results_for_math_vectors/LogiQA/math_vector/*/*/*")[0],
}

def grade(d):
    """Re-grade every arm identically from raw predictions, so no stored
    metrics.json (some written by the superseded grader) can leak in."""
    rows = [json.loads(l) for l in open(f"{d}/predictions.jsonl")]
    out = []
    for r in rows:
        gen = r["model_generation"][0]
        pred = extract_choice(gen, options=options_from_prompt(r.get("prompt", "")))
        out.append({
            "correct": pred is not None and pred == int(r["answer"]),
            "answered": pred is not None,
        })
    return out

def mcnemar(a, b, key):
    only_a = sum(1 for x, y in zip(a, b) if x[key] and not y[key])
    only_b = sum(1 for x, y in zip(a, b) if y[key] and not x[key])
    n = only_a + only_b
    chi2 = (abs(only_a - only_b) - 1) ** 2 / n if n else 0.0
    p = math.erfc(math.sqrt(chi2 / 2)) if n else 1.0
    return only_a, only_b, chi2, p

data = {k: grade(v) for k, v in ARMS.items()}
print(f"{'arm':<10}{'n':>5}{'acc':>8}{'answered':>10}{'answer_rate':>13}")
print("-" * 46)
for k, v in data.items():
    n = len(v); c = sum(x["correct"] for x in v); a = sum(x["answered"] for x in v)
    print(f"{k:<10}{n:>5}{c/n:>8.3f}{a:>10}{a/n:>13.3f}")

print(f"\n{'comparison':<24}{'A>B':>6}{'B>A':>6}{'net':>7}{'chi2':>9}{'p':>10}  verdict")
print("-" * 78)
pairs = [("baseline","v_logic"),("baseline","v_code"),("baseline","v_math"),
         ("v_logic","v_code"),("v_logic","v_math"),("v_code","v_math")]
for x, y in pairs:
    for key, lbl in (("correct","acc"), ("answered","ans")):
        oa, ob, chi2, p = mcnemar(data[x], data[y], key)
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"{x}->{y} ({lbl}){'':<{max(0,10-len(x)-len(y))}}{oa:>6}{ob:>6}{ob-oa:>+7}{chi2:>9.2f}{p:>10.4f}  {sig}")
