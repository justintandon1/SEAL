"""GPU-free smoke test for layer_analysis.py.

Fabricates hidden.pt files in the exact format hidden_analysis.py writes
(list indexed by hidden-state layer; per trace: step states + check/switch
indices), with a planted execution-vs-non-execution separation that GROWS
with layer depth. If layer_analysis.py works, its probe accuracy must
recover that gradient (near-chance at layer 1, near-perfect at layer 28).

Run:  python test_layer_analysis.py
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile

import torch

LAYERS = [1, 5, 10, 15, 20, 25, 28]
NUM_HIDDEN_STATES = 29   # embeddings + 28 blocks, as for the 1.5B model
DIM = 64
N_TRACES = 30
STEPS_PER_TRACE = 20


def make_hidden_pt(save_dir, seed):
    # One shared planted direction across both trace pools (as in real data,
    # where correct/incorrect traces share the model's geometry); only the
    # noise differs per pool.
    g_dir = torch.Generator().manual_seed(1234)
    direction = torch.randn(DIM, generator=g_dir)
    direction /= direction.norm()
    g = torch.Generator().manual_seed(seed)

    hidden_dict = [{} for _ in range(NUM_HIDDEN_STATES)]
    for k in range(N_TRACES):
        n = STEPS_PER_TRACE
        perm = torch.randperm(n, generator=g)
        check_index = perm[:4]          # ~20% reflection
        switch_index = perm[4:6]        # ~10% transition
        non_exec = torch.zeros(n, dtype=torch.bool)
        non_exec[check_index] = True
        non_exec[switch_index] = True

        for layer in LAYERS:
            # separation strength grows linearly with depth
            shift = 4.0 * layer / (NUM_HIDDEN_STATES - 1)
            h = torch.randn(n, DIM, generator=g)
            h[non_exec] += shift * direction
            hidden_dict[layer][k] = {
                "step": h,
                "check_index": check_index.clone(),
                "switch_index": switch_index.clone(),
            }
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, os.path.join(save_dir, "hidden.pt"))


def main():
    data_dir = tempfile.mkdtemp(prefix="seal_layer_analysis_test_")
    try:
        prefixs = [f"correct_0_{N_TRACES}", f"incorrect_0_{N_TRACES}"]
        for i, p in enumerate(prefixs):
            make_hidden_pt(os.path.join(data_dir, f"hidden_{p}"), seed=i)

        cmd = [sys.executable, "layer_analysis.py",
               "--data_dir", data_dir,
               "--prefixs", *prefixs,
               "--layers", *[str(l) for l in LAYERS],
               "--max_per_class", "400",
               "--seed", "0"]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))

        out_dir = os.path.join(data_dir, "layer_analysis")
        for f in ["fig3_replication.png", "layer_metrics.png", "layer_metrics.csv"]:
            path = os.path.join(out_dir, f)
            assert os.path.exists(path), f"missing output: {f}"
            print(f"OK  {f}  ({os.path.getsize(path)} bytes)")

        with open(os.path.join(out_dir, "layer_metrics.csv")) as f:
            rows = {int(r["layer"]): r for r in csv.DictReader(f)}
        acc_first = float(rows[LAYERS[0]]["probe_acc"])
        acc_last = float(rows[LAYERS[-1]]["probe_acc"])
        print(f"probe accuracy: layer {LAYERS[0]} = {acc_first:.3f}, "
              f"layer {LAYERS[-1]} = {acc_last:.3f}")
        assert acc_last > 0.9, "planted signal at deep layer not recovered"
        assert acc_last > acc_first + 0.15, "no depth gradient recovered"
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
