#!/usr/bin/env python3
"""
Visualize a SEAL baseline-vs-steered run to reproduce the paper's headline:
steering keeps (or improves) accuracy while cutting reasoning-token usage.

Reads each run's `math_eval.jsonl` (written by get_math_results.py) and produces:
  - accuracy_comparison.png   accuracy: baseline vs steered
  - token_usage.png           avg total + "thinking" tokens, with % reduction
  - token_distribution.png    per-example token-length distributions
  - dashboard.png             combined 2x2 summary (good for slides)
  - summary.json / summary.md  numbers + a presentable table

Use from the command line:
  python visualize_results.py --baseline_dir ... --steered_dir ... \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output_dir .../figures

...or inline in a notebook:
  import visualize_results as vr
  summary, fig = vr.visualize(baseline_dir, steered_dir,
                              model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                              dataset="GSM8K", output_dir="figures")  # fig shows inline

Token counts use the model tokenizer when available; otherwise it falls back to
whitespace word counts (axes are labelled accordingly).
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

BASELINE_COLOR = "#9aa0a6"
STEERED_COLOR = "#1a73e8"
THINK_COLOR = "#f9ab00"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_eval_file(path):
    """Accept a file or a dir; return the graded eval file with the most rows.

    Tries math_eval.jsonl (MATH/GSM runs), then code_eval.jsonl (APPS/MBPP/
    LiveCodeBench runs -- same `model_generation` + `all_eval` shape), then
    falls back to predictions.jsonl (lengths only, no accuracy).
    """
    if os.path.isfile(path):
        return path
    for pattern in ("math_eval.jsonl", "code_eval.jsonl", "predictions.jsonl"):
        candidates = glob.glob(os.path.join(path, "**", pattern), recursive=True)
        if candidates:
            return max(candidates, key=lambda p: sum(1 for _ in open(p)))
    raise FileNotFoundError(f"No math_eval.jsonl / code_eval.jsonl / predictions.jsonl under {path}")


def load_run(path):
    f = find_eval_file(path)
    rows = [json.loads(l) for l in open(f)]
    print(f"  loaded {len(rows):>5} rows  <-  {f}")
    return rows, f


# --------------------------------------------------------------------------- #
# Tokenizer (optional) + measurement
# --------------------------------------------------------------------------- #
def get_token_counter(model):
    if not model:
        return None, "words"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        return (lambda t: len(tok.encode(t, add_special_tokens=False))), "tokens"
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] tokenizer unavailable ({e}); falling back to word counts")
        return None, "words"


def think_slice(text):
    """Return the <think>...</think> span if present, else the whole text."""
    if "<think>" in text:
        body = text.split("<think>", 1)[1]
        return body.split("</think>", 1)[0] if "</think>" in body else body
    return text


def measure(rows, counter):
    """Return (correct: bool array, total_len array, think_len array)."""
    count = counter if counter else (lambda t: len(t.split()))
    correct, total_len, think_len = [], [], []
    for r in rows:
        gens = r.get("model_generation") or [r.get("model_output", "")]
        idx = r.get("mv_index", 0)
        idx = idx if 0 <= idx < len(gens) else 0
        gen = gens[idx] if gens else ""
        total_len.append(count(gen))
        think_len.append(count(think_slice(gen)))
        if "mv_eval" in r:
            correct.append(bool(r["mv_eval"]))
        elif "all_eval" in r and r["all_eval"]:
            correct.append(bool(r["all_eval"][idx if idx < len(r["all_eval"]) else 0]))
    return np.array(correct, dtype=bool), np.array(total_len), np.array(think_len)


# --------------------------------------------------------------------------- #
# Analysis bundle
# --------------------------------------------------------------------------- #
def analyze(baseline_dir, steered_dir, model=None, dataset="GSM8K"):
    """Load both runs and compute everything the plots need. Returns a dict."""
    counter, unit = get_token_counter(model)
    print("Loading baseline...")
    base_rows, _ = load_run(baseline_dir)
    print("Loading steered...")
    steer_rows, _ = load_run(steered_dir)

    base_ok, base_tot, base_thk = measure(base_rows, counter)
    steer_ok, steer_tot, steer_thk = measure(steer_rows, counter)
    base_acc = base_ok.mean() if base_ok.size else float("nan")
    steer_acc = steer_ok.mean() if steer_ok.size else float("nan")

    summary = {
        "dataset": dataset, "model": model, "unit": unit,
        "baseline": {"n": len(base_rows), "accuracy": float(base_acc),
                     "avg_total": float(base_tot.mean()),
                     "median_total": float(np.median(base_tot)),
                     "avg_thinking": float(base_thk.mean())},
        "steered": {"n": len(steer_rows), "accuracy": float(steer_acc),
                    "avg_total": float(steer_tot.mean()),
                    "median_total": float(np.median(steer_tot)),
                    "avg_thinking": float(steer_thk.mean())},
        "delta": {"accuracy_pts": float((steer_acc - base_acc) * 100),
                  "total_reduction_pct": float((1 - steer_tot.mean() / base_tot.mean()) * 100),
                  "thinking_reduction_pct": float((1 - steer_thk.mean() / base_thk.mean()) * 100)},
    }
    return {"summary": summary, "unit": unit, "dataset": dataset, "model": model,
            "base": {"acc": base_acc, "tot": base_tot, "thk": base_thk},
            "steer": {"acc": steer_acc, "tot": steer_tot, "thk": steer_thk}}


def summary_markdown(a):
    s, unit = a["summary"], a["unit"]
    b, st, d = s["baseline"], s["steered"], s["delta"]
    return "\n".join([
        f"# SEAL reproduction — {a['dataset']}",
        "",
        f"Model: `{a['model'] or 'n/a'}`  ·  length unit: **{unit}**",
        "",
        "| Metric | Baseline | Steered | Δ |",
        "|---|---|---|---|",
        f"| Accuracy | {b['accuracy']*100:.1f}% | {st['accuracy']*100:.1f}% | {d['accuracy_pts']:+.1f} pts |",
        f"| Avg total {unit} | {b['avg_total']:.0f} | {st['avg_total']:.0f} | {-d['total_reduction_pct']:+.0f}% |",
        f"| Avg thinking {unit} | {b['avg_thinking']:.0f} | {st['avg_thinking']:.0f} | {-d['thinking_reduction_pct']:+.0f}% |",
        f"| Median total {unit} | {b['median_total']:.0f} | {st['median_total']:.0f} | |",
        "",
    ])


# --------------------------------------------------------------------------- #
# Plots (each takes an Axes so they compose into the dashboard)
# --------------------------------------------------------------------------- #
def _bar_labels(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")


def plot_accuracy(ax, a):
    accs = [a["base"]["acc"] * 100, a["steer"]["acc"] * 100]
    bars = ax.bar(["baseline", "steered"], accs, color=[BASELINE_COLOR, STEERED_COLOR])
    _bar_labels(ax, bars, "{:.1f}%")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy")
    ax.set_ylim(0, max(100, max(accs) * 1.15))


def plot_tokens(ax, a):
    unit = a["unit"]
    bt, st = a["base"]["tot"], a["steer"]["tot"]
    bk, sk = a["base"]["thk"], a["steer"]["thk"]
    x = np.arange(2); w = 0.38
    b1 = ax.bar(x - w / 2, [bt.mean(), st.mean()], w, label=f"total {unit}", color=STEERED_COLOR)
    b2 = ax.bar(x + w / 2, [bk.mean(), sk.mean()], w, label=f"thinking {unit}", color=THINK_COLOR)
    _bar_labels(ax, b1, "{:.0f}"); _bar_labels(ax, b2, "{:.0f}")
    ax.set_xticks(x); ax.set_xticklabels(["baseline", "steered"])
    ax.set_ylabel(f"avg {unit} / problem")
    red = (1 - st.mean() / bt.mean()) * 100 if bt.mean() else 0
    ax.set_title(f"Generation length  ({-red:+.0f}% total)"); ax.legend(fontsize=9)


def plot_distribution(ax, a):
    unit = a["unit"]; bt, st = a["base"]["tot"], a["steer"]["tot"]
    hi = np.percentile(np.concatenate([bt, st]), 99)
    bins = np.linspace(0, max(hi, 1), 40)
    ax.hist(bt, bins=bins, alpha=0.55, color=BASELINE_COLOR, label=f"baseline (med {np.median(bt):.0f})")
    ax.hist(st, bins=bins, alpha=0.65, color=STEERED_COLOR, label=f"steered (med {np.median(st):.0f})")
    ax.axvline(bt.mean(), color=BASELINE_COLOR, ls="--", lw=2)
    ax.axvline(st.mean(), color=STEERED_COLOR, ls="--", lw=2)
    ax.set_xlabel(f"{unit} / problem"); ax.set_ylabel("# problems")
    ax.set_title("Length distribution"); ax.legend(fontsize=9)


def plot_efficiency(ax, a):
    unit = a["unit"]
    bx, by = a["base"]["tot"].mean(), a["base"]["acc"] * 100
    sx, sy = a["steer"]["tot"].mean(), a["steer"]["acc"] * 100
    ax.scatter(bx, by, s=160, color=BASELINE_COLOR, zorder=3)
    ax.scatter(sx, sy, s=160, color=STEERED_COLOR, zorder=3)
    ax.annotate("baseline", (bx, by), textcoords="offset points", xytext=(6, 6))
    ax.annotate("steered", (sx, sy), textcoords="offset points", xytext=(6, 6))
    ax.annotate("", xy=(sx, sy), xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color="#34a853", lw=2))
    ax.set_xlabel(f"avg {unit} / problem  (lower = cheaper)")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Efficiency frontier (up-left is better)")


def make_dashboard(a):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_accuracy(axes[0, 0], a)
    plot_tokens(axes[0, 1], a)
    plot_distribution(axes[1, 0], a)
    plot_efficiency(axes[1, 1], a)
    fig.suptitle(f"SEAL on {a['dataset']} — {a['model'] or ''}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_all(a, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(a["summary"], f, indent=2)
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write(summary_markdown(a))
    for name, plot in [("accuracy_comparison", plot_accuracy),
                       ("token_usage", plot_tokens),
                       ("token_distribution", plot_distribution)]:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot(ax, a); fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150); plt.close(fig)
    fig = make_dashboard(a)
    fig.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150)
    return fig


def visualize(baseline_dir, steered_dir, model=None, dataset="GSM8K",
              output_dir=None, show=True):
    """Notebook-friendly entry point. Returns (summary_dict, dashboard_figure)."""
    a = analyze(baseline_dir, steered_dir, model=model, dataset=dataset)
    fig = save_all(a, output_dir) if output_dir else make_dashboard(a)
    print(summary_markdown(a))
    if output_dir:
        print(f"Wrote figures + summary to {output_dir}")
    return a["summary"], fig


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--steered_dir", required=True)
    ap.add_argument("--model", default=None, help="HF model id for the tokenizer (optional)")
    ap.add_argument("--dataset", default="GSM8K")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    plt.switch_backend("Agg")  # headless for the CLI
    visualize(args.baseline_dir, args.steered_dir, model=args.model,
              dataset=args.dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
