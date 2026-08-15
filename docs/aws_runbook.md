# SEAL steering on AWS EC2 — runbook

Written for the R_iso control arms (`docs/random_vector_build_plan.html`), but
any steered eval runs the same way.

## Pick the box

The steered evals do **not** use vLLM — `eval_MATH_steering.py` loads the custom
`Qwen2ForCausalLM` through HF `generate()` in **fp32** (no `torch_dtype` is
passed). So VRAM is set by fp32 weights + fp32 KV cache, not by the 1.5B
parameter count:

| | fp32 |
|---|---|
| weights | 6.6 GB |
| KV cache, batch 25 × 10k tokens | 13.8 GB |
| **peak** | **~20–21 GB** |

24 GB is enough — the Vast runbook ran this config on a 4090. 48 GB buys
headroom and speed.

| Instance | GPU | $/hr* | Per arm | Spot |
|---|---|---|---|---|
| `g6e.xlarge` | L40S 48GB, 864 GB/s | ~1.86 | ~11 h | **~$8** |
| `g5.xlarge` | A10G 24GB, 600 GB/s | ~1.01 | ~13 h | ~$5 |
| `g6.xlarge` | L4 24GB, 300 GB/s | ~0.81 | ~15 h | ~$5 |

\* us-east-1 on-demand, approximate — check current rates. Spot cuts ~60%.

**Budget ~11 hours per arm, not the ~2 h this table used to claim.** The seed-1
arm took **10h53m** on `g6e.xlarge`. The old estimate assumed the run was
memory-bandwidth bound; it is not (see below), so bandwidth-based scaling
between GPUs does not predict much either.

**Don't use the P family.** AWS has no single-GPU A100/H100; `p4d.24xlarge` is
8× A100 at ~$32/hr. And an A100 would not shorten an arm anyway — see below.

**Don't switch to bf16 to save memory.** The baseline and the five SEAL arms
were generated in fp32. Changing dtype changes greedy decoding paths, and the
paired McNemar test assumes the arms differ *only* in the steering vector. Same
reason to leave `batch_size` at 25.

## What actually governs speed

The seed-1 arm ran at **~196 ms per decode step** at batch 25. For a 1.5B model
that is roughly an order of magnitude off what the hardware can do, and it says
the run is **Python/kernel-launch bound** in this repo's eager-mode
`modeling_qwen2.py` — not memory-bandwidth bound.

So a faster GPU buys very little: most of the wall clock is CPU-side overhead
between kernels, which an A100 does not fix. Pay for a cheap 24–48 GB card and
spend the savings on spot. The real fix would be `torch.compile` or CUDA graphs,
but both change numerics and so are off the table mid-study.

## 0. Request quota first — this is the blocker

New accounts have **0 vCPUs** for G instances. Service Quotas → EC2 →
*Running On-Demand G and VT instances* → request **8+ vCPUs**
(`g6e.xlarge` = 4; ask for 16 if you might run seeds in parallel).

Spot is a **separate quota**: *All G and VT Spot Instance Requests*. Request it
at the same time — with checkpointing, spot is the default choice here.

Approval takes hours to a couple of days. Do this before anything else.

## 1. Launch

- **AMI**: *Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)* — drivers
  and CUDA only. Prefer it over the framework DLAMIs: `requirements.txt` pins
  `vllm==0.6.6.post1` → torch 2.5.1, and a preinstalled torch just gets
  overwritten.
- **Instance**: `g6e.xlarge` (or `g5.xlarge`).
- **Storage**: root gp3, **200 GB**. The default is too small once the HF cache
  and results land.
- **Key pair**: create one, download the `.pem`, `chmod 400` it.
- **Security group**: SSH (22) from **your IP only**, not 0.0.0.0/0.
- **Region**: `g6e` is not in every region; us-east-1 and us-west-2 are safe.

If you take spot, set *Interruption behavior* to **Stop**, not Terminate — the
EBS volume survives, so a resumed run still has its checkpoint.

## 2. Connect and verify the GPU

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>
nvidia-smi          # confirm the GPU and >= 45 GiB (or >= 22 GiB on g5)
```

If `nvidia-smi` fails the AMI is wrong — stop and relaunch rather than
installing drivers by hand.

## 3. Clone and install

```bash
git clone https://github.com/Andwwy/SEAL.git
cd SEAL
cp .env.example .env      # HF_TOKEN optional: the 1.5B model is public
bash setup_runpod.sh      # despite the name, it is generic
```

`setup_runpod.sh` installs `requirements.txt`, installs the **vendored**
`latex2sympy2` with `--no-deps` (the PyPI one breaks on import and pins an
incompatible antlr), and runs import/GPU checks. Takes 5–15 min.

Set `HF_HOME` somewhere on the big volume if the root fills:

```bash
export HF_HOME=$HOME/hf_cache
```

## 4. Preflight

```bash
python scripts/verify_random_vectors.py
```

All checks must pass — the run script gates on this, so a bad vector fails in
seconds instead of OOMing two hours in.

## 5. Run, inside tmux

**Use tmux.** A dropped SSH connection kills the run otherwise, and each arm is
~11 hours.

```bash
tmux new -s seal

# MATH-500, single GPU
SEEDS="2 3" bash scripts/run_random_control.sh 0

# LogiQA clean-500, single GPU
DATASETS=logiqa SEEDS="1 2 3" bash scripts/run_random_control.sh 0

# either one across every GPU on the box
DATASET=logiqa scripts/run_arm_multigpu.sh results/control/R_iso_seed1.pt R_iso_seed1

# detach: Ctrl-b then d      reattach: tmux attach -t seal
```

tqdm ticks once per batch, 20 batches per arm, ~33 min each. A batch much over
~40 min means something is wrong — check `nvidia-smi` for the model landing on
CPU. Sharded runs print per shard into `<save_dir>/logs/rank<N>.log`.

**Budget LogiQA like MATH: ~11 h per arm.** The median LogiQA generation is only
848 tokens, but 181 of the 500 hit the 10,000-token cap — and a batch runs until
*all* 25 sequences finish. The chance a batch of 25 contains no capped sequence
is about 1 in 60,000, so effectively every batch runs the full 10,000 steps.

## Multi-GPU

```bash
DATASET=logiqa scripts/run_arm_multigpu.sh results/control/R_iso_seed1.pt R_iso_seed1
```

One process per visible GPU, pinned with `CUDA_VISIBLE_DEVICES`, then merge and
score. `NUM_GPUS` overrides the count. Batches are dealt out round-robin as
whole units, so an example keeps the same neighbours — and the same padding —
it would have had on one GPU.

**Do not use torchrun/DDP/FSDP.** There is no gradient and nothing to
all-reduce; the shards never talk. Plain background processes have strictly
fewer failure modes. The same scheme works across *nodes* — give each node a
disjoint set of `--shard_rank` values against shared storage.

**Do not let `device_map="auto"` see more than one GPU.** With 8 visible it
spreads a 6.6 GB model across all of them as a pipeline, running one GPU at a
time plus cross-device copies — slower than a single GPU. The launcher pins each
shard to one device, which is what prevents this.

An A100 is still not the answer (see above); more GPUs are. `g6e.12xlarge`
(4× L40S, ~$10.49/hr, ~2.8 h/arm) is the best value; `p4d.24xlarge` costs more,
finishes no faster, and needs a separate 96-vCPU *P instance* quota.

## Checkpointing and resume

Every batch is appended to `predictions.partial.jsonl` in the results directory
(`predictions.partial.NofM.jsonl` when sharded) and `fsync`ed before the next
`generate()` call. At this decode rate a batch is ~33 minutes, so a crash, an
OOM, or a spot reclaim costs one batch — not eleven hours.

**To resume, re-run the exact same command.** Finished batches are skipped; the
run prints how many examples it recovered and how many batches remain. Nothing
else is needed — no flag, no separate resume mode.

This is what makes **spot** worth taking: ~60% off, and a reclaim is now a
nuisance rather than a lost day. Pair it with *Interruption behavior: Stop*.

Two guards, both there to stop a silently-wrong arm rather than to catch bugs:

- `run_config.json` records model, `steering_layer`, coef, `max_tokens`,
  `batch_size`, seed and vector path. A resume with any of these changed is
  refused, by field name. `steering_layer`, `max_tokens` and `batch_size` are
  **not** encoded in the results path, so this is the only thing standing
  between you and two half-arms merged into one file.
- Scoring refuses to run on an incomplete set. A partial arm is worse than no
  arm, because the accuracy still looks plausible.

Resume is at **batch** granularity, never mid-batch. Batches are left-padded to
their longest member, so an example's neighbours are part of its numerics;
redoing the whole batch keeps every example in the same company it had in the
interrupted run. Splitting a batch across two runs would repack it and break the
assumption the paired McNemar test rests on — the same class of error as
reusing a baseline with a different `SAMPLE_SEED`.

A killed process usually leaves a torn final line. That is expected and handled:
the line is dropped and its batch redone.

Resume works across a *different* GPU count, since every checkpoint file in the
directory is read, not just the current shard's. To merge by hand (e.g. after
collecting shards from several nodes):

```bash
python scripts/merge_shards.py --save_dir <leaf dir> --dataset LogiQA
```

## Running the control on LogiQA

```bash
DATASETS=logiqa SEEDS="1 2 3" bash scripts/run_random_control.sh 0
```

LogiQA is scored on the contamination-free explicit selection
(`data/LogiQA/eval_rand42_500_clean.json`), **not** the seeded sample that
MATH-500 uses. That is what every existing LogiQA arm ran, and the wiring is
already set in `run_random_control.sh` — don't override `LOGIQA_SELECTION`
unless you intend to break the pairing.

The existing baseline at
`results_for_logic_vectors/LogiQA/baseline/base_remove_bos/rand42_500_clean` is
reusable: verified to be the same 500 problems in the same order as the SEAL
arms, at `MAX_TOKENS=10000` (181 sequences capped at exactly 10,000, matching
its `n_unfinished`). The directory name differs from the arms'
(`rand42_500_clean` vs `eval_rand42_500_clean`) for historical reasons; the
contents match.

**Read a LogiQA null with the effect size in mind.** Baseline is 0.258 against
0.284–0.298 for the SEAL vectors — 13 to 20 problems out of 500, against 75 on
MATH-500. A control arm there has much less room to separate "does nothing" from
"reproduces the effect," so a null is correspondingly weaker evidence.

## 6. Collect results, then stop

```bash
# from your laptop
rsync -avz -e "ssh -i ~/.ssh/your-key.pem" \
  ubuntu@<public-ip>:~/SEAL/results/results_for_control_vectors/ \
  ./results/results_for_control_vectors/

python scripts/report_stats.py --benchmark math500
```

Commit only the four artifact files per arm — `example_prompt.txt`,
`predictions.jsonl`, `math_eval.jsonl`, `metrics.json` — as the seed-1 arm did.
`predictions.partial.jsonl` and `run_config.json` are run scaffolding; `results/`
is gitignored, so they only reach a commit if you `git add -f` the whole
directory.

Then **stop the instance**. Idle `g6e.xlarge`:

| Left running | Cost |
|---|---|
| 1 day | $45 |
| 1 week | $313 — more than $200 of credits |

Set a billing alarm at $25 before launching. *Stop* keeps the EBS volume (~$16/mo
for 200 GB); *terminate* deletes everything.

## Gotchas

- **Batches are padded.** HF `generate()` runs a batch until *all* 25 sequences
  finish, so a batch costs its longest member. With a 32.8% cap rate, ~100% of
  batches run the full 10,000 steps — steering's token savings buy almost no
  wall-clock.
- **Reusing the baseline** (`RUN_BASELINE=0`) is only valid while
  `SAMPLE_SEED=42`, `MATH_MAX_EXAMPLES=500` and `MAX_TOKENS=10000` match the
  existing arm. The paired McNemar test depends on identical problem ordering,
  and a mismatch produces a number that looks fine and means nothing.
- **Parallel seeds**: two instances, same total cost, ~11 h instead of ~22.
  Needs 8 vCPUs of quota.
- **`rsync` mid-run is safe.** The checkpoint is append-only and fsynced per
  batch, so pulling partial results to check progress cannot corrupt it.
- **Read-only HF token** if you set one at all — never a write token on a rented
  box.
