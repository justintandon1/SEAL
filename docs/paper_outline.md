# Paper outline — reasoning calibration across domains

**Target:** workshop, 4–8 pages (assume ~7 + appendix)

**Lead claim:** reasoning calibration is a *single domain-general direction*. Vectors
extracted by the SEAL recipe from math, code, and logic traces are geometrically
near-identical and behaviourally indistinguishable — pooling and combining them change
nothing. Calibration is not a per-skill object the way task/content directions are.

**Supporting claims, in order of strength:**

| # | Claim | Evidence |
|---|---|---|
| 1 | Geometry predicts behaviour | 86.7% of domain-vector variance in one singular direction; cos(S_general, S_combo) = 0.981; then all 10 pairwise McNemar tests non-significant on all three fully-covered benchmarks |
| 2 | Direction is load-bearing, not magnitude | Norm-matched isotropic random vector at the same site/coef moves the *opposite* way: tokens +9.5% (MATH) / +17.5% (LogiQA), cap rate up, accuracy down |

---

## Research questions (pre-registered)

From the [v2 research plan](steering_seal_research_plan.html), stated before any results.
Restate these in §1 and answer them explicitly in §4 — pre-registration is a real
strength here and should be visible, not implied.

> **Primary RQ.** When a single general reasoning-calibration vector is extracted (SEAL
> recipe) from a mixture of reasoning tasks, how much of each task-specific vector's
> accuracy and token-efficiency gain does it recover — and is the general vector a linear
> combination of the task-specific vectors, or a distinct direction?

| | Question as stated | Answer | Status |
|---|---|---|---|
| **Q1** Recovery | How much of each task-specific vector's home-benchmark gain — accuracy *and* token reduction — does the pooled general vector recover, and at what cost? | ~100%, but the question is degenerate: there is nothing to recover from. S_general is within 0.6 pts of every domain vector on MATH-500, 0.2–0.4 on LogiQA, all non-significant | Answered |
| **Q2** Geometry | Is the general vector well explained by the span of the three task-specific vectors, or does a non-trivial residual remain? | Well explained. cos(S_general, S_combo) = 0.981; domain vectors are near rank-1 (86.7% of variance in one singular direction). Report ‖r‖/‖S_general‖ to close it formally | Answered — needs the residual number computed |
| **Q3** Specificity | (a) Do task-specific vectors help their home benchmark while degrading others? (b) Does negating a vector (−S) re-introduce over-thinking? | (a) **No.** S_code is the best vector on LogiQA; S_math beats S_code on MBPP. No home-benchmark advantage anywhere. (b) **Unrun** | (a) answered, (b) open |

**The pre-registered hypothesis was confirmed.** The plan predicted "S_general ≈ S_combo
with a small residual — establishing that the reasoning-calibration direction is largely
domain-general, in contrast to skill/content directions which tend to be domain-specific."
That is exactly what the geometry and the effect matrix show. Say so in §1: a confirmed
advance prediction is much stronger than the same finding reported post-hoc.

**Two deviations from the plan to state plainly in §3, not bury:**

| Plan specified | Actually run | Consequence |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-**7B** | **1.5B** for all five vectors and all six benchmarks; 7B has S_math on MATH-500 only | Describe the 7B row as a scale check, not as the main model |
| Traces from GSM8K / MBPP / LogiQA | **MATH train / APPS train / LogiQA** | Extraction corpora differ from the plan; GSM8K and MBPP became *evaluation* benchmarks instead. Harmless, but the paper must describe what was done |

**Q3(b) is the cheapest open item on the list.** Negating a committed vector costs no
build time and one eval arm, and the plan names it as the control that shows the
direction is meaningful. If −S lengthens the CoT and drops accuracy, it also
independently corroborates claim 2 — and R_iso already showed the site responds to a
wrong-pointing vector, so the expected outcome is clean. Worth slotting beside the
coefficient sweep.

---

## §1 Introduction (~0.75 p)

| Beat | Content |
|---|---|
| Problem | Reasoning models overthink; SEAL calibrates this with one training-free direction |
| Open question | Inherited from task arithmetic / Steer2Adapt: is a *general* capability direction the combination of *domain* directions? Prior work assumes yes; nobody tests it |
| What we do | Five vectors under one fixed recipe (3 domain, 1 pooled, 1 combined), 6 benchmarks, plus a magnitude-matched random control |
| Finding | The question dissolves. There is one direction. Recovery is ~100% because there is nothing to recover from |
| Contributions | (i) domain-generality result with matched extraction; (ii) geometry that predicts it a priori, and a magnitude-matched random control showing the direction is what carries it |

## §2 Background (~1 p — three subsections, compressed for workshop length)

| Sub | Content | Purpose in the argument |
|---|---|---|
| 2.1 | **Overthinking and activation steering.** Reasoning models (R1-distill family) emit long self-verifying CoT; efficiency literature attacks this directly (budget forcing, concise-CoT, early exit). Activation steering as the alternative: one direction, one layer, added at decode, no retraining — CAA / ITI / refusal-direction lineage. Primitive: `h ← h + α·S` | Establishes the problem and defines the object. ~2 paragraphs; setup, not contribution. Merged from what would be two sections at full length |
| 2.2 | **SEAL's calibration recipe.** The contrast at `\n\n` boundaries inside `<think>`, the keyword taxonomy (reflection / transition / execution) from SEAL Appendix B, the defaults (layer 20, α = −1.0) | State explicitly that we hold this recipe *fixed* and vary only the source corpus. That is the methodological precondition for the domain-generality claim, so it belongs here, not in Methods |
| 2.3 | **Do interventions compose?** Task arithmetic (Ilharco et al. 2022): in weight space, adding task vectors adds skills, negating removes them. Steer2Adapt: composes steering vectors in activation space, treats a general capability as a combination of domain directions — but assumes the composition works. ATLAS: extraction should match evaluation (traces, not role descriptions) | The pivot. Steer2Adapt is the foil — it assumes what we test. ATLAS justifies our extraction design |
| 2.4 | **Gap statement** (1 para, closes §2). Nobody has tested whether a *calibration* direction — as opposed to a skill or content direction — decomposes by domain at all, and prior compositional work assumes rather than checks that domain directions are the right unit | Sets up the headline claim. Keep it to the composition question — the budget/truncation issue is now a limitation, not a gap we claim to fill |

> **At full-conference length** this becomes five subsections, splitting 2.1 and adding a
> dedicated "controls and confounds in steering evaluation" section. At workshop length
> that confound material lives in §3.4 and §5 instead.

## §3 Experimental design (~1.5 p)

| Sub | Content | Notes |
|---|---|---|
| 3.1 | **Setup.** One table: model DeepSeek-R1-Distill-Qwen-1.5B (1536-d); site = residual stream, layer 20, all generated positions, BOS excluded; intervention `h ← h + coef·S` at coef −1.0; decoding vLLM greedy (T=0), n=1, max_tokens 10,000; pairing = identical problems in identical order across every arm (verified) | Layer 20 and α = −1.0 are **SEAL's published defaults, adopted unchanged** — not parameters we selected. Say this explicitly: the operating point is part of the recipe being held fixed (§2.2), the same as the boundary definition and keyword taxonomy. The study varies the source corpus and nothing else |
| 3.2 | **Five vectors, one recipe.** Four steps: generate traces (500 correct + 500 incorrect, so it is not a correctness direction) → segment at `\n\n` inside `<think>` → tag by keyword → contrast the piles. Table of source corpus, trace counts, RT boundaries, execution boundaries, ‖S‖ | See sign-convention note and per-vector notes below |
| 3.3 | **Geometry analysis.** Pairwise cosine matrix (5×5); SVD of stacked domain vectors, singular values [1.613, 0.541, 0.325], top direction carries 86.7% of squared variance; least-squares projection of S_general onto span{S_math, S_code, S_logic} reporting ‖r‖/‖S_general‖; cos(S_general, S_combo) = 0.981 | Costs no GPU time and *predicts* the benchmark result — present in that order. The 0.981 is the single most direct pooling-vs-combining test |
| 3.4 | **Benchmarks, metrics, statistics.** Six benchmarks: MATH-500, LogiQA (clean English-500), APPS (rand42-500), MBPP-500, GSM8K (1,319), MMLU philosophy (311); full 5-vector coverage on the first three. Graders: Omni-MATH rule grader / executed unit tests / reasoning extractor. Metrics: accuracy, avg + median tokens (model tokenizer), cap-hit rate. Statistics: exact two-sided McNemar on discordant pairs | Report cap-hit rate alongside tokens for every arm — it is cheap, and it is what makes the §5 budget limitation checkable rather than asserted. Disclose the MATH extractor's no-answer fallback (credits 2 truncated baseline generations); note correcting it *strengthens* the effect and we keep the conservative number. Justify McNemar over independent CIs in one sentence: arms agree on 86–93% of problems, unpaired tests discard that. No multiple-comparison correction, stated as the conservative choice |
| 3.5 | **Controls.** `R_iso` — isotropic random, rescaled to 57.06, three pinned seeds, identical site and coefficient. Answers: does any perturbation of this magnitude reproduce the effect? State the pre-registered interpretation grid | Scope out the null-keyword (B) and shuffled-label (C) controls **explicitly**: they answer reasoning-*specificity*, a different claim than domain-generality. Naming them and saying why they are not required here is stronger than omitting them |

**Sign convention — state `S` and `α` together, never apart.** The code stores
`S = mean(H_reflection ∪ H_transition) − mean(H_execution)` and applies it at
`coef = −1.0`; the net intervention is therefore `+(H_E − H_RT)`, pushing the residual
stream toward the execution mean. Use the code convention in Methods so every number,
`meta.json`, and script flag in the released artifacts matches the paper without
translation — then add one plain interpretive sentence immediately after. Verified in
`vector_generation.py:47`, `build_general_vector.py:298`, and `build_combo_vector.py:130`
(which rejects inputs using the opposite convention). Note the v2 research plan uses the
*other* convention (`E − RT` with `+α`); the two are equivalent but must not be mixed.
Open item: check which convention SEAL's own paper uses.

**Per-vector notes for §3.2**

| Vector | Source | Traces | RT bounds | Exec bounds | ‖S‖ |
|---|---|---|---|---|---|
| S_math | MATH train | 500 + 500 | 51,317 | 101,743 | 59.67 |
| S_code | codeparrot/apps train | 500 + 500 | 76,452 | 113,249 | 56.22 |
| S_logic | LogiQA 2.0 MRC, English | 500 + 500 | 9,683 | 13,680 | 55.29 |
| S_general | pooled raw activations, phase 1 | — | 137,452 | 228,672 | 55.78 |
| S_combo | unit-sum of the three, rescaled | — | — | — | 57.06 |

- **S_general** pools raw boundary activations across all three domains into one contrast.
  Phase 1 is count-weighted, so MATH+APPS supply 93.6% of rows and LogiQA 6.4%. Report
  this as a *design fact here*, not as a caveat later — it is why S_general sits at
  cos 0.797 from S_logic. Phase 2 (domain-balanced) is implemented but unrun → limitations.
- **S_combo** unit-normalizes each source, equal weights, rescales to mean-source norm
  57.06. Justify equal weights in one sentence: sources are near rank-1, so across a 4×
  weight range the direction stays within cos 0.985 — grid search would resolve ~0.015 of
  cosine.

## §4 Results (~1.5 p)

| Sub | Content | Figure/table |
|---|---|---|
| 4.1 | Geometry | Cosine heatmap + SVD spectrum |
| 4.2 | Effect matrix — 5 vectors × 3 benchmarks, accuracy and tokens | Main table |
| 4.3 | Indistinguishability — pairwise McNemar; 376/500 unanimity on MATH-500, 410/500 on APPS. Emphasise the *flatness of the columns*, not who leads | Pairwise table (one benchmark in body, rest to appendix) |
| 4.4 | Random control — R_iso does not reproduce the effect and inverts the token change | R_iso vs baseline vs S_combo |

## §5 Limitations (~0.75 p) — write it, do not bury it

**The budget paragraph — write it plainly, it is the one a reviewer will test.** All runs
cap generation at 10,000 tokens, and 33–84% of baseline generations reach that cap
depending on benchmark. Our MATH-500 baseline of 64.4% sits below the commonly cited
83.9% for this model because that figure uses a 32k budget with sampling; ours is greedy
at 10k. Because a capped generation states no answer and scores zero, a large share of
the measured accuracy change on every benchmark comes from previously-capped problems
rather than from problems the model could already finish — on MATH-500 the 164 capped
problems move ~0% → 51.2% while the 336 finished ones move 95.2% → 93.2%. The 7B model,
with a much higher baseline and less truncation, shows a correspondingly smaller effect
(+3.4). The 7B numbers make this concrete rather than suggestive: it truncates on
**13.0%** of MATH-500 against the 1.5B's 32.8% — roughly a quarter the truncation — and
returns roughly a quarter the effect, with the same shape. Its 65 capped problems move
6.2% → 53.8% (contributing +6.2) while its 435 finished ones move 96.1% → 92.9%
(−2.8), netting +3.4. We therefore report our results as holding **at this operating
point**, and do not claim the intervention improves reasoning quality independent of the
budget. Cap-hit rates are reported for every arm (§3.4) so this is checkable.

**On the operating point.** Layer 20 and α = −1.0 are SEAL's published defaults, adopted
unchanged and held constant across every arm including the controls. We do not sweep them:
the study's design is to fix the recipe and vary only the source corpus, and the
coefficient is part of that recipe. Magnitude comparability across the five vectors is
handled by construction rather than by tuning — the domain vectors span ‖S‖ 55.29–59.67,
S_general lands at 55.78, and S_combo is rescaled to the mean source norm 57.06, so no
vector is advantaged by strength at a shared coefficient. The consequence to state is
narrow: our results characterise these vectors at SEAL's operating point, not the best
achievable effect for any of them.

Also: single greedy sample, no sampling variance. Coverage holes (S_general/S_combo on
MBPP/GSM8K/MMLU). S_general is Phase 1 only. Reasoning-specificity untested (Controls B/C).
Scale: the 7B arm covers one vector on one benchmark, so it supports "the intervention
transfers to 7B" but not the domain-generality claim itself (see the 7B section below).

## §6 Conclusion (~0.25 p)

---

## Appendix

A. Full coverage matrix · B. Per-benchmark pairwise McNemar tables · C. S_combo build
derivation + SVD · D. Ablation plan (budget check, Phase-2 balanced S_general, rank-1 direction steering) ·
E. Compute costs.

---

## Evidence status

**Have it:** geometry; MATH-500 / LogiQA / APPS at full 5-vector coverage; MBPP / GSM8K /
MMLU partial; all pairwise stats; the decomposition on all six; R_iso seed 1 on MATH-500
and LogiQA; 7B S_math on MATH-500.

**Still needed, ranked:**

| # | Item | Why | Cost |
|---|---|---|---|
| 1 | R_iso seed 1 on APPS, then seeds 2–3 | Serves claim 2 directly. Without ≥3 seeds the rank-sum framing does not apply and the control is n=1 | 2–3 GPU-h + APPS arm |
| 2 | Anti-vector (−S) arm on MATH-500 — Q3(b) | The one pre-registered question with no answer at all. No build cost, one eval arm | 1 eval arm |
| 3 | Coverage holes; Phase-2 S_general | Acceptable as limitations | varies |

**Deferred — not in this submission:** the 32k budget sweep. It is the right experiment
and we say so, but it is out of scope for this deadline. Consequence: the §5 budget
paragraph must carry the caveat squarely rather than promising a future check, and
"future work" should name it explicitly so a reviewer sees it was considered, not missed.

**No longer blocking under this framing:** Control B (null-keyword) and the shuffled-label
control. They test reasoning-specificity, which this paper does not claim.

---

## Scale: the 7B gap

**What exists.** One vector, one benchmark.

| Asset | State |
|---|---|
| S_math_7b (dim 3584, ‖S‖ 82.44) | built, committed |
| MATH-500 baseline + S_math_7b | run: 84.4% → 87.8%, **+3.4, McNemar p = 0.030**, agreement 89.0% (403 both-correct, 42 both-wrong, b = 19, c = 36) |
| Tokens (7B tokenizer) | baseline 3,380 avg / 2,055 median / 13.0% at cap → steered 2,776 / 1,782 / 8.6%, **−17.9%** — same shorten-and-improve pattern as 1.5B |
| Cap decomposition | 65 capped: 6.2% → 53.8% (**+6.2**); 435 finished: 96.1% → 92.9% (**−2.8**); net +3.4 |
| R_iso_7b seeds 1–3 (norm-matched 82.44) | **built and verified, never evaluated** |
| S_code_7b, S_logic_7b, S_general_7b, S_combo_7b | do not exist |
| Any 7B benchmark other than MATH-500 | none |

**The problem for the paper.** The headline claim is that vectors from different domains
are indistinguishable. *One vector cannot test that.* S_math_7b alone shows "SEAL works at
7B," which is SEAL's result, not ours. As it stands 7B is a sanity check, not a
replication — and a reviewer of a 1.5B-only claim will ask about scale.

**Ranked 7B plan, by value per GPU-hour.**

| # | Item | Buys | Cost |
|---|---|---|---|
| ~~1~~ | ~~Recompute 7B token counts~~ — **done**. `scripts/report_stats.py` now registers a `math500_7b` benchmark with a per-benchmark tokenizer; run `python scripts/report_stats.py --benchmark math500_7b` | Closed the stated limitation and supplied the §5 decomposition above | done |
| 2 | Build **S_logic_7b** (cheapest second domain: short CoTs, existing `gen_logiqa_vllm.py` pipeline) | The 7B **cosine matrix**. If the 7B domain vectors are also near rank-1, claim 1's *geometric* half replicates at scale **with no eval arms at all** | trace gen + extraction |
| 3 | One eval arm: S_logic_7b on MATH-500 (reuse existing 7B baseline) | Tests indistinguishability at 7B — does a logic-derived vector match a math-derived one on MATH, as at 1.5B? Directly replicates the headline | 1 eval arm |
| 4 | R_iso_7b_seed1 on MATH-500 | Replicates claim 2 at scale. Vectors already built and verified — pure eval cost | 1 eval arm |

**The key move is #2.** Splitting the replication into a *geometric* half (needs vectors
only) and a *behavioural* half (needs eval arms) means the cheap half can land even if GPU
budget runs out. A 7B cosine matrix showing the same near-rank-1 structure is a real scale
result on its own, and it costs no evaluation.
