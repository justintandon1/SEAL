# Archived — LogiQA 300, mixed-language eval set

Moved here from `results/results_for_math_vectors/LogiQA/` on 2026-07-29. Kept for
provenance, not for reporting. Superseded by the English-only, contamination-free
500-problem set.

## Why archived

These runs scored 300 problems drawn from the raw `datatune/LogiQA2.0` test split.
That split ships an untranslated Chinese mirror alongside the English MRC rows, so
roughly half the problems are Chinese — while the steering vectors, and the keyword
tagging that builds them (`hidden_analysis.py` matches English phrases like "wait",
"alternatively"), are English-only. The eval set and the vector were not measuring
the same thing.

The set also predates the contamination check: LogiQA2.0's own train and test splits
overlap, and nothing here screens for it.

## What replaced it

`data/LogiQA/eval_rand42_500_clean.json` — 500 English-only problems with the 16
items that also appear in the extraction set swapped out. Results:
`results/results_for_logic_vectors/LogiQA/*/rand42_500_clean/`.

The two are not comparable: different size (300 vs 500), language mix, contamination
status, and baseline (0.327 vs 0.258).

The numbers on this page are all from the pre-2026-07-29 extractor, which guessed a
letter for generations that never closed `</think>`. They are kept as a record of what
these superseded runs reported, not as measurements — nothing here has been regraded,
and nothing here should be compared against a current run.

## What is here

| Run | Vector | acc |
|---|---|---|
| `transfer_baseline/rand42_300` | none | 0.327 |
| `transfer_steered/baseline_10000_vector_500_500_.../rand42_300` | S_math | 0.367 |

The matching S_code run on this same set (acc 0.400) lives in the `v_code-SEAL`
repo under `eval_result/results_for_code_vectors/LogiQA/`, and is archived there
separately if at all.

`visualize_all.py` and `scripts/build_steering_examples_html.py` still read from
this directory so their existing figures reproduce; both are labelled archived at
the reference site.
