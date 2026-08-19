# 7B math vector → LogiQA (legacy seed-42 500)

Cross-domain transfer: Akhilesh’s 7B `S_math` (`layer_20_transition_reflection_steervec.pt`)
on LogiQA 2, n=500, seed 42, layer 20, coef −1.0, `max_tokens=10000`, chat,
`--remove_bos`.

**This is not the clean-500.** The selection is
`data/LogiQA/eval_rand42_500.json` (16 train overlaps). Do **not** McNemar-pair
these rows with `logiqa_7b` / `eval_rand42_500_clean.json` (PR #69 S_logic).
`scripts/report_stats.py` registers this run as `logiqa_7b_legacy`.

| Arm | Acc | Answer rate | Unfinished |
|---|---:|---:|---:|
| Baseline | 0.488 | 0.758 | 115 |
| S_math steered | 0.528 | 0.852 | 62 |
| Δ | +4.0 pp | +9.4 pp | −53 |
