# 7B S_combo evals (MATH-500 / LogiQA / APPS)

Steering vector: packed `vectors/S_combo_7b.pt` (dim 3584, fp32, ‖S‖ ≈ 85.04,
SHA256 `2695da60794adea4ecfacd99b79ada30775dbb6ac89b81d210a3219e20ebaff0`).
Equal-weight unit-norm sum of 7B S_math / S_code / S_logic, `--norm mean-source`.
Model `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, layer 20, coef −1.0,
`max_tokens=10000`, chat, `--remove_bos`, n=500 seed 42.

Write-up: `docs/steering_transfer_results_7b_combo.html`.
`scripts/report_stats.py` registers `math500_7b_s_combo`, `logiqa_7b_s_combo`,
and `apps_7b_s_combo`. Steered arms live in this tree. Unsteered baselines are
the folders under `results_for_7b_logic_vectors/…/baseline/` — do **not** copy
them here, and do **not** McNemar-pair MATH against
`results_for_7b_math_vectors/MATH500/` (Akhilesh 84.4% baseline, different job)
or LogiQA against `results_for_7b_math_vectors/LogiQA/` (legacy dirty 500).

| Bench | Baseline | S_combo | Δ |
|---|---:|---:|---:|
| MATH-500 | 0.850 | 0.896 | +4.6 |
| LogiQA clean-500 | 0.490 | 0.524 | +3.4 |
| APPS test rand42-500 | 0.284 | 0.302 | +1.8 |
