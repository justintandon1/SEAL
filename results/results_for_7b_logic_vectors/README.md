# 7B S_logic evals (LogiQA / MATH-500 / APPS)

Steering vector: packed `logiqa2_v_logic.pt` (dim 3584, fp32, ‖S‖ ≈ 93.86,
SHA256 `cfb5556a13d1b442fb88f60379f4ffc60e067e638daee5256d50a92ff1c2e83e`).
Model `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, layer 20, coef −1.0,
`max_tokens=10000`, chat, `--remove_bos`, n=500 seed 42.

Write-up: `docs/steering_transfer_results_7b_logic.html`.
`scripts/report_stats.py` registers `logiqa_7b`, `math500_7b_s_logic`, and
`apps_7b_s_logic`. Do **not** McNemar-pair MATH against
`results_for_7b_math_vectors/MATH500/` (Akhilesh 84.4% baseline, different job)
or LogiQA against `results_for_7b_math_vectors/LogiQA/` (legacy dirty 500).

| Bench | Baseline | S_logic | Δ |
|---|---:|---:|---:|
| MATH-500 | 0.850 | 0.876 | +2.6 |
| LogiQA clean-500 | 0.490 | 0.486 | −0.4 |
| APPS test rand42-500 | 0.284 | 0.326 | +4.2 |
