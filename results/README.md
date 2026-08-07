# Results layout

Results are grouped by the steering vector they were produced with, so multiple
contributors can add runs without clobbering each other.

```
results/
  results_for_math_vectors/     # runs using the MATH-derived steering vector (S_math)
    GSM/                        # math benchmark (GSM8K)
    MATH500/                    # math benchmark (MATH-500)
    MBPP/                       # code benchmark
    APPS/                       # code benchmark
    LiveCodeBench/              # code benchmark
    MMLU/                       # knowledge benchmark
    MATH_train/                 # vector-extraction artifacts
    GSM_MBPP_LogiQA/            # cross-domain summary (all_domains.png, summary.md)
    MATH500_APPS_LogiQA/        # cross-domain summary (math_apps_logiqa.png, summary.json)

  results_for_logic_vectors/    # runs using the LogiQA-derived steering vector (S_logic)
    LogiQA/                     # logic benchmark — in-domain
    MATH500/                    # math benchmark — transfer
    APPS/                       # code benchmark — transfer
    LogiQA_train/               # vector-extraction artifacts

  results_for_code_vectors/     # runs using the APPS-derived code vector (S_code = vectors/apps_v_code.pt)
    MATH500/ APPS/ GSM/ MBPP/   # imported from Andwwy/v_code-SEAL (see its README)
    LogiQA/                     # in-repo run on the clean English-500 set

  results_for_general_vectors/  # runs using the pooled general vector (S_general, Phase 1)
    MATH500/ LogiQA/            #   <benchmark>/s_general_phase1/<run>/  + visualization/

  results_for_combo_vectors/    # runs using the combined vector (S_combo = results/general/S_combo.pt)
    MATH500/ LogiQA/ APPS/      #   <benchmark>/s_combo/<run>/  + visualization/

  archive/
    logiqa_300_mixed_language/  # superseded 300-problem, ~half-Chinese LogiQA set
```

`S_general` and `S_combo` are both *general* vectors but are built differently, so
they get separate trees: `S_general` pools raw boundary activations across all
three domains, while `S_combo` is a weighted sum of the three finished domain
vectors. See `docs/s_combo_construction.html` and `docs/s_general_vs_s_combo.html`.

`hidden.pt` files are not committed in either tree — only `hidden_*/prompts.json`
and the resulting `vector_*/…steervec.pt`. Regenerate hidden states from the
build scripts when needed.

Cross-domain summary folders are named after the datasets they combine
(e.g. `GSM_MBPP_LogiQA/`), never a generic name like `combined/`.

## Adding your own results

If you steer with a different vector, create a sibling folder named
`results_for_<source>_vectors/` (e.g. `results_for_code_vectors/`) and keep the
same per-benchmark subfolder structure. Point your visualization scripts at your
own root via the `RESULTS_ROOT` constant (see `visualize_all.py`).
