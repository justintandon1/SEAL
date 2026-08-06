"""LogiQA 2.0 support for the SEAL MATH-style eval scripts (Harness A).

Lets `eval_MATH_vllm.py` (baseline) and `eval_MATH_steering.py` (steered) target
LogiQA with the *same* generation + steering core they use for MATH/GSM. LogiQA is
the same "shape" as GSM (reason, then a short answer) — only the answer is
multiple-choice, so scoring is choice accuracy instead of numeric equality.

  - load:    datatune/LogiQA2.0 (config "default", split "test"), STREAMED — a
             non-streaming load materializes the 63k-row train split onto the
             (slow) HF cache and hangs.
  - prompt:  a chat instruction that elicits a <think> block + a final
             'Answer: X' (so the SEAL steering trigger on `\\n\\n` inside <think>
             still fires).
  - grade:   `logic_eval_main` mirrors get_math_results.main's output schema
             (all_pred / all_eval / mv_pred / mv_eval / mv_index + math_eval.jsonl
             + metrics.json), so visualize_results.py works unchanged.
"""
import json
import os
import re
from collections import Counter


def is_cjk_heavy(ex, threshold=0.1):
    """True if a LogiQA row is mostly Chinese script.

    LogiQA 2.0 ships the untranslated Chinese source alongside the English
    MRC rows, so any split is roughly half CJK. Canonical definition lives
    here so the build set (gen_logiqa_vllm.py) and the eval set
    (load_logiqa(english_only=True)) filter identically.
    """
    text = ex["passage"] + " " + ex["question"] + " " + " ".join(ex["options"])
    if not text:
        return False
    cjk = sum(1 for c in text if ord(c) > 0x2E80)
    return cjk / len(text) > threshold


def load_logiqa(split="test", config="default", english_only=False, cjk_threshold=0.1):
    from datasets import load_dataset

    ds = load_dataset("datatune/LogiQA2.0", config, split=split, streaming=True)
    out, skipped = [], 0
    for r in ds:
        # LogiQA 2.0 mixes MRC rows (text/question/options/answer — what we want)
        # with NLI rows (major_premise/conclusion/label) and a few malformed JSON
        # lines. Skip anything that isn't a valid MRC row.
        try:
            d = json.loads(r["text"])
            out.append({
                "passage": d["text"],
                "question": d["question"],
                "options": d["options"],
                "gt": int(d["answer"]),
            })
        except (KeyError, ValueError, TypeError):
            skipped += 1
    if skipped:
        print(f"[logiqa] skipped {skipped} non-MRC or malformed rows")
    if english_only:
        before = len(out)
        out = [ex for ex in out if not is_cjk_heavy(ex, cjk_threshold)]
        print(f"[logiqa] english_only: kept {len(out)}/{before} rows "
              f"({before - len(out)} skipped as CJK-heavy, threshold={cjk_threshold})")
    return out


LOGIQA_INSTRUCTION = (
    "Read the passage and answer the multiple-choice question. "
    "Think step-by-step, then end your response with 'Answer: X' where X is A, B, C, or D.\n\n"
    "Passage: {passage}\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}"
)


def build_logiqa_prompt(ex):
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(ex["options"]))
    return LOGIQA_INSTRUCTION.format(
        passage=ex["passage"].strip(),
        question=ex["question"].strip(),
        options=opts,
    )


_ANSWER_RE = re.compile(r"answer\s*(?:is|:)?\s*\(?([ABCD])(?![A-Za-z])\)?", re.IGNORECASE)
_LOGIEVAL_HEAD = re.compile(r"^\s*\(?([ABCD])(?![A-Za-z])\)?", re.IGNORECASE)
_OPTION_LINE = re.compile(r"^\s*([ABCD])[.)]\s+(.*\S)\s*$", re.MULTILINE)


def options_from_prompt(prompt):
    """Recover the option strings from a rendered LogiQA/MMLU prompt.

    predictions.jsonl does not carry the options list, but build_logiqa_prompt
    renders them as 'A. <text>' lines, so they can be read back for the
    native-schema match below.
    """
    block = prompt.rsplit("Options:", 1)[-1] if "Options:" in prompt else prompt
    return {m.group(2).strip(): "ABCD".index(m.group(1).upper())
            for m in _OPTION_LINE.finditer(block)}


def extract_choice(gen, options=None):
    """Answer extraction adapted to reasoning-model output.

    No upstream harness covers this case. LogiQA 2.0's own baselines score a
    classification head (no generation at all), and LogiEval -- the authors'
    generative eval, upstreamed to lm-eval-harness as the `logieval` task --
    assumes the model answers immediately, so it anchors at position 0
    (`regex: "^\\s*([A-D])"` then `take_first`, metric `exact_match`). A model
    that emits thousands of `<think>` tokens first matches that anchor never.

    Kept from LogiEval:
      * exactly one extraction attempt, no guessing
      * NO fallback -- a generation that does not state an answer is wrong,
        not assigned a letter
    Adapted for reasoning output:
      * the answer region is what follows `</think>`, not the start of the text
      * within that region take the LAST match, since our prompt asks the model
        to *end* with 'Answer: X' (LogiEval takes the first because its prompt
        puts the answer first)

    Returns an option index (the dataset's native label type -- LogiQA 2.0 ships
    `answer` as an int 0-3 and `options` as an unlabelled list; A/B/C/D exists
    only in the prompt rendering) or None when the model never answered.

    Passing `options` (text -> index) enables a verbatim option match as a
    secondary path. That returns an index directly and cannot invent an answer,
    unlike a bare-letter guess.
    """
    if "</think>" not in gen:
        return None  # never left the reasoning block -> no answer was given
    tail = gen.split("</think>", 1)[1]

    m = _ANSWER_RE.findall(tail)
    if m:
        return "ABCD".index(m[-1].upper())

    head = _LOGIEVAL_HEAD.match(tail)  # LogiEval's own pattern, applied to our answer region
    if head:
        return "ABCD".index(head.group(1).upper())

    if options:
        stripped = tail.strip()
        for text, idx in sorted(options.items(), key=lambda kv: -len(kv[0])):
            if stripped.endswith(text) or stripped == text:
                return idx
    return None


def logic_eval_main(res_path, save=False, output_dir=None):
    """Grade LogiQA/MMLU predictions, keeping get_math_results.main's output schema
    so visualize_results.py works unchanged.

    A generation that never closed </think> is UNFINISHED: the model ran out of
    budget still inside the reasoning block, so it never reached the answer region
    and no answer exists to read. Unfinished scores as incorrect -- matching
    LogiEval's exact_match semantics, where a generation that does not state an
    answer is simply wrong -- but is counted separately so "reasoned and got it
    wrong" is distinguishable from "never stopped reasoning".

    There is deliberately no way to select a different grader. The pre-2026-07-29
    extractor guessed a letter for unfinished generations, inflating LogiQA by
    ~10 points; keeping it selectable is what let a legacy-graded baseline sit on
    main for a week while every steered arm used this rule. metrics.json still
    records "grader": "reasoning" so a file's provenance stays checkable.
    """
    with open(res_path) as f:
        data = [json.loads(line) for line in f]

    for example in data:
        gens = example.get("model_generation") or [example.get("model_output", "")]
        gt = int(example["answer"])
        opts = options_from_prompt(example.get("prompt", ""))
        all_pred = [extract_choice(g, options=opts) for g in gens]
        all_eval = [(p is not None and p == gt) for p in all_pred]
        all_unfinished = ["</think>" not in g for g in gens]

        valid = [p for p in all_pred if p is not None]
        if valid:
            pred = Counter(valid).most_common(1)[0][0]
            index = all_pred.index(pred)
        else:
            pred, index = None, 0

        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = bool(all_eval[index]) if all_eval else False
        example["mv_index"] = index
        # Unfinished only when NO sampled generation terminated; with n=1 that is
        # just "this generation never closed </think>".
        example["unfinished"] = all(all_unfinished)
        example["answered"] = pred is not None

    n = len(data)
    acc = sum(e["mv_eval"] for e in data) / n if n else 0.0
    n_unfinished = sum(e["unfinished"] for e in data)
    n_answered = sum(e["answered"] for e in data)
    acc_answered = (sum(e["mv_eval"] for e in data) / n_answered) if n_answered else 0.0

    metrics = {
        "acc": acc,                                   # unfinished counts as incorrect
        "n": n,
        "n_answered": n_answered,
        "n_unfinished": n_unfinished,
        "answer_rate": n_answered / n if n else 0.0,  # did the model terminate and answer
        "acc_answered": acc_answered,                 # accuracy among those that answered
        "grader": "reasoning",
    }
    print(f"Accuracy: {acc:.3f}  "
          f"(answered {n_answered}/{n} = {metrics['answer_rate']:.3f}, "
          f"unfinished {n_unfinished}, acc|answered {acc_answered:.3f})")

    if save:
        with open(os.path.join(output_dir, "math_eval.jsonl"), "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

    return acc
