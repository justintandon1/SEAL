"""Replicate SEAL Figure 3 (t-SNE of thought activations per layer) plus
quantitative separation metrics the paper omits.

Consumes the hidden.pt files written by hidden_analysis.py (all layers, or any
subset extracted with --keep_layers). Everything here is CPU-only.

Per layer, computes:
  - t-SNE panel of execution / reflection(check) / transition(switch) states
    (the Figure 3 replication)
  - linear-probe accuracy (logistic regression, execution vs non-execution,
    held-out split) — quantitative version of "separable in latent space"
  - silhouette score (execution vs non-execution, cosine distance)
  - ||mean(H_check ∪ H_switch) − mean(H_other)|| — the steering-vector norm.
    Raw norms grow with depth, so a fixed coef injects harder at deeper
    layers; reporting this alongside accuracy addresses the norm confound
    the paper's Figure 6 ignores (cf. general_vector_steering's
    constant-norm sweep).

Usage (after extraction with e.g. --keep_layers 1 5 10 15 20 25 28):
  python layer_analysis.py \
      --data_dir results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
      --prefixs correct_0_500 incorrect_0_500 \
      --layers 1 5 10 15 20 25 28

Outputs (under <data_dir>/layer_analysis/):
  fig3_replication.png   t-SNE grid, one panel per layer
  layer_metrics.csv      probe acc / silhouette / vector norm per layer
  layer_metrics.png      metrics vs layer
"""
import argparse
import csv
import os

import numpy as np
import torch

from vector_generation import load_data

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import train_test_split
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def subsample(x, n, rng):
    if x.shape[0] <= n:
        return x
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx]


def layer_arrays(check, switch, other, layer, max_per_class, rng):
    """Return (execution, reflection, transition) float32 numpy arrays."""
    refl = subsample(check[layer], max_per_class, rng).float().numpy()
    tran = subsample(switch[layer], max_per_class, rng).float().numpy()
    exe = subsample(other[layer], max_per_class, rng).float().numpy()
    return exe, refl, tran


def probe_accuracy(exe, non_exe, rng, seed):
    """Held-out accuracy of a linear probe separating execution vs non-execution.

    Classes are balanced by downsampling so 50% is always chance level.
    """
    n = min(len(exe), len(non_exe))
    x = np.concatenate([
        exe[rng.choice(len(exe), n, replace=False)],
        non_exe[rng.choice(len(non_exe), n, replace=False)],
    ])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(xtr, ytr)
    return float(clf.score(xte, yte))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing hidden_<prefix>/hidden.pt")
    parser.add_argument("--prefixs", type=str, nargs="+",
                        default=["correct_0_500", "incorrect_0_500"])
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[1, 5, 10, 15, 20, 25, 28],
                        help="Hidden-state indices to analyze (paper grid). Must "
                             "be present in hidden.pt (check --keep_layers used "
                             "at extraction).")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap traces read per prefix (RAM control).")
    parser.add_argument("--max_per_class", type=int, default=1500,
                        help="Subsample cap per thought class per layer.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Default: <data_dir>/layer_analysis")
    args = parser.parse_args()

    if not HAVE_SKLEARN:
        raise SystemExit("scikit-learn is required: pip install scikit-learn")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir or os.path.join(args.data_dir, "layer_analysis")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"loading hidden states for layers {args.layers} ...")
    check, switch, other = load_data(args.data_dir, args.prefixs, args.layers,
                                     max_examples=args.max_examples)
    for l in args.layers:
        print(f"  layer {l}: check={check[l].shape[0]} switch={switch[l].shape[0]} "
              f"other={other[l].shape[0]}")

    # Paper-matching palette: execution purple, reflection blue, transition pink
    colors = {"Execution": "#9b8bb4", "Reflection": "#b8d4ee", "Transition": "#f4b8c1"}

    n_panels = len(args.layers)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.2))
    if n_panels == 1:
        axes = [axes]

    rows = []
    for ax, layer in zip(axes, args.layers):
        exe, refl, tran = layer_arrays(check, switch, other, layer,
                                       args.max_per_class, rng)
        non_exe = np.concatenate([refl, tran])

        acc = probe_accuracy(exe, non_exe, rng, args.seed)

        sil_x = np.concatenate([subsample(exe, 1000, rng), subsample(non_exe, 1000, rng)])
        sil_y = np.concatenate([np.zeros(min(len(exe), 1000)), np.ones(min(len(non_exe), 1000))])
        sil = float(silhouette_score(sil_x, sil_y, metric="cosine"))

        # Same contrast as vector_generation.py, so this is the norm of the
        # steering vector that would be built at this layer.
        vec = np.concatenate([refl, tran]).mean(0) - exe.mean(0)
        vec_norm = float(np.linalg.norm(vec))

        rows.append({"layer": layer, "probe_acc": acc, "silhouette": sil,
                     "vector_norm": vec_norm,
                     "n_check": int(check[layer].shape[0]),
                     "n_switch": int(switch[layer].shape[0]),
                     "n_other": int(other[layer].shape[0])})
        print(f"layer {layer}: probe_acc={acc:.3f} silhouette={sil:.3f} "
              f"||S||={vec_norm:.2f}")

        emb = TSNE(n_components=2, random_state=args.seed, init="pca",
                   perplexity=30).fit_transform(np.concatenate([exe, refl, tran]))
        splits = np.cumsum([len(exe), len(refl)])
        for name, pts in zip(["Execution", "Reflection", "Transition"],
                             np.split(emb, splits)):
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5,
                       color=colors[name], label=name)
        ax.set_title(f"Layer {layer}  (probe {acc:.0%})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Thought-type activations at \\n\\n boundaries (SEAL Fig. 3 replication)")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(out_dir, "fig3_replication.png"), dpi=150)
    plt.close(fig)

    csv_path = os.path.join(out_dir, "layer_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    layers = [r["layer"] for r in rows]
    for ax, key, label in zip(
            axes,
            ["probe_acc", "silhouette", "vector_norm"],
            ["Linear-probe accuracy\n(execution vs non-execution)",
             "Silhouette score (cosine)",
             "Steering-vector norm ||S||"]):
        ax.plot(layers, [r[key] for r in rows], "o-")
        ax.set_xlabel("layer"); ax.set_title(label, fontsize=10); ax.grid(alpha=0.3)
    axes[0].axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_metrics.png"), dpi=150)
    plt.close(fig)

    print(f"\nwrote {out_dir}/fig3_replication.png, layer_metrics.png, layer_metrics.csv")


if __name__ == "__main__":
    main()
