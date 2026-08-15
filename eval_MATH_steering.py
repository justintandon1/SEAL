import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import  AutoTokenizer
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from peft import PeftModel, PeftConfig

import sys
import os
import gc
import glob
from tqdm import tqdm

from get_math_results import main as eval_main
from logic_utils import load_logiqa, build_logiqa_prompt, logic_eval_main
from mmlu_utils import load_mmlu, build_mmlu_prompt
os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a


def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans


# Batch-level checkpoint, so a crash 10 hours into a run costs one batch rather
# than the whole arm. Written alongside the final predictions.jsonl and kept
# after a successful run, so re-grading never means re-generating.
CHECKPOINT_GLOB = "predictions.partial*.jsonl"

# Settings a resume has to agree on. Ones already encoded in save_dir (vector
# name, coef) are listed too so the check stands alone; the ones that are not --
# steering_layer, max_tokens, batch_size, model -- are the whole point, since
# nothing else would catch two half-arms merged into one file.
FINGERPRINT_KEYS = [
    "model_name_or_path", "dataset", "mmlu_subject", "sample_seed",
    "random_sample", "max_examples", "start", "max_tokens", "batch_size",
    "use_chat_format", "remove_bos", "steering", "steering_vector",
    "steering_layer", "steering_coef", "logiqa_english_only",
    "logiqa_eval_selection",
]


def checkpoint_path(save_dir, shard_rank, num_shards):
    if num_shards == 1:
        return os.path.join(save_dir, "predictions.partial.jsonl")
    return os.path.join(save_dir, f"predictions.partial.{shard_rank}of{num_shards}.jsonl")


def _read_checkpoint(path):
    """Parse one checkpoint file, dropping a torn final line from a killed run."""
    records = []
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in rec:
                records.append(rec)
    return records


def load_completed(save_dir):
    """Example index -> record, across every checkpoint file in save_dir.

    Reading *all* files rather than only this process's own is what lets a run
    resume on a different GPU count: the batch-to-shard assignment changes, but
    a batch any earlier process finished is still skipped.
    """
    completed = {}
    for path in sorted(glob.glob(os.path.join(save_dir, CHECKPOINT_GLOB))):
        for rec in _read_checkpoint(path):
            completed[rec["idx"]] = rec
    return completed


def check_or_write_fingerprint(args, n_examples):
    """Refuse to resume a checkpoint written under different settings."""
    path = os.path.join(args.save_dir, "run_config.json")
    current = {k: getattr(args, k) for k in FINGERPRINT_KEYS}
    current["n_examples"] = n_examples

    if os.path.exists(path):
        with open(path) as fin:
            previous = json.load(fin)
        drift = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
        if drift:
            raise SystemExit(
                f"Refusing to resume: {path} was written under a different config.\n"
                + "\n".join(f"  {k}: checkpoint={p!r} now={n!r}" for k, (p, n) in sorted(drift.items()))
                + "\nDelete the directory to start clean, or fix the arguments."
            )
        return

    # Atomic, so a concurrent shard never reads a half-written file.
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fout:
        json.dump(current, fout, indent=2, sort_keys=True)
    os.replace(tmp, path)


def finalize(save_dir, dataset):
    """Merge the checkpoints into predictions.jsonl, then score."""
    completed = load_completed(save_dir)
    with open(os.path.join(save_dir, "run_config.json")) as fin:
        n = json.load(fin)["n_examples"]

    missing = [i for i in range(n) if i not in completed]
    if missing:
        # A partial arm is worse than no arm: the accuracy still looks plausible.
        raise SystemExit(
            f"Incomplete: {len(missing)}/{n} predictions missing (first: {missing[:5]}). "
            "Re-run the same command -- finished batches are skipped."
        )

    out = os.path.join(save_dir, "predictions.jsonl")
    with open(out, "w") as fout:
        for i in range(n):
            rec = dict(completed[i])
            rec.pop("idx")  # keep the artifact identical to pre-checkpoint runs
            fout.write(json.dumps(rec) + "\n")
    print(f"Merged {n} predictions -> {out}")

    if dataset in ("LogiQA", "MMLU"):
        logic_eval_main(out, save=True, output_dir=save_dir)
    else:
        eval_main(out, save=True, k=None, output_dir=save_dir)


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            gt = extract_box(example["solution"])
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt":gt,
            })
    elif args.dataset == "GSM":
        data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer":example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    elif args.dataset == "LogiQA":
        for ex in load_logiqa(english_only=args.logiqa_english_only):
            test_data.append({
                "question": ex["question"],
                "passage": ex["passage"],
                "options": ex["options"],
                "answer": ex["gt"],
                "gt": ex["gt"],
            })
    elif args.dataset == "MMLU":
        for ex in load_mmlu(subject=args.mmlu_subject):
            test_data.append({
                "question": ex["question"],
                "options": ex["options"],
                "answer": ex["gt"],
                "gt": ex["gt"],
            })
    else:
        raise ValueError("Dataset not supported")

    if args.logiqa_eval_selection:
        # Explicit pool indices override the seeded sample. Indices are positions
        # in load_logiqa(english_only=True) order, i.e. rows of data/LogiQA/test.jsonl.
        # Used for eval_rand42_500_clean.json (contamination-free) and for scoring
        # just the replacement items.
        with open(args.logiqa_eval_selection) as fin:
            sel = json.load(fin)
        idxs = [item["pool_idx"] for item in sel["items"]]
        bad = [i for i in idxs if i >= len(test_data)]
        if bad:
            raise ValueError(f"selection has out-of-range pool_idx (pool size {len(test_data)}): {bad[:5]}")
        test_data = [test_data[i] for i in idxs]
        print(f"[eval] explicit selection: {len(test_data)} items from {args.logiqa_eval_selection}")
    elif args.random_sample:
        if args.max_examples and len(test_data) > args.max_examples:
            # Seeded random sample over the full split instead of the first N in
            # file order. Same scheme as v_code-SEAL eval/benchmarks.py (shuffle
            # indices, keep sorted first N) so both harnesses score identical tasks.
            idx = list(range(len(test_data)))
            random.Random(args.sample_seed).shuffle(idx)
            test_data = [test_data[i] for i in sorted(idx[:args.max_examples])]
    else:
        if args.start:
            test_data = test_data[args.start:]
        if args.max_examples and len(test_data) > args.max_examples:
            test_data = test_data[:args.max_examples]

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    prompts = []
    for i, example in enumerate(test_data):
        if args.dataset == "LogiQA":
            content = build_logiqa_prompt(example)
        elif args.dataset == "MMLU":
            content = build_mmlu_prompt(example)
        else:
            content = prefix + "Question: " + example["question"].strip()
        prompt = content + "\nAnswer: "
        if args.use_chat_format:
            if args.dataset in ("LogiQA", "MMLU") or "deepseek" in args.model_name_or_path:
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    if args.shard_rank == 0:
        with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            fout.write(prompts[0])

    check_or_write_fingerprint(args, len(prompts))


    # The batch is the unit of both work and recovery. That is not just because
    # generate() produces one at a time: batches are left-padded to their
    # longest member, so an example's neighbours are part of its numerics.
    # Keeping batch boundaries at fixed global offsets -- across shards, and
    # across a resume -- means every example is scored in the same company it
    # would have had in a single uninterrupted run. The paired McNemar test
    # depends on that, the same way it depends on identical problem ordering.
    batch_starts = list(range(0, len(prompts), args.batch_size))
    my_starts = [s for j, s in enumerate(batch_starts)
                 if j % args.num_shards == args.shard_rank]

    completed = load_completed(args.save_dir)
    todo = [s for s in my_starts
            if not all(i in completed for i in range(s, min(s + args.batch_size, len(prompts))))]

    tag = "" if args.num_shards == 1 else f"[shard {args.shard_rank}/{args.num_shards}] "
    print(f"{tag}{len(my_starts)} batches assigned, "
          f"{len(my_starts) - len(todo)} already checkpointed, {len(todo)} to run")

    checkpoint = checkpoint_path(args.save_dir, args.shard_rank, args.num_shards)
    # Rewrite our own file from its parsed records, so a torn trailing line from
    # a killed run cannot corrupt the lines about to be appended.
    if os.path.exists(checkpoint):
        mine = _read_checkpoint(checkpoint)
        with open(checkpoint, "w") as fout:
            for rec in mine:
                fout.write(json.dumps(rec) + "\n")

    if not todo:
        print(f"{tag}nothing to generate")
        return

    if "qwen" in args.model_name_or_path.lower():
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    else:
        raise ValueError("Model not supported")

    if args.steering:
        steer_vec = torch.load(args.steering_vector, weights_only=True)
        steer_vec = steer_vec.to(model.device)
        model.set_steering_flag(steering_flag=True, steering_layer=args.steering_layer, steer_vec=steer_vec,  steer_coef=args.steering_coef, tokenizer=tokenizer)

    with open(checkpoint, "a") as fout:
        for s in tqdm(todo, desc=tag.strip() or None):
            if args.steering:
                model.start_new_round()
            batch = prompts[s:s+args.batch_size]
            tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
            tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
            with torch.no_grad():
                output = model.generate(**tokenized_batch, do_sample=False, max_new_tokens=args.max_tokens,use_cache=True)
            prompt_len = tokenized_batch["input_ids"].shape[1]
            output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]

            for offset, text in enumerate(output):
                i = s + offset
                fout.write(json.dumps({
                    "idx": i,
                    "prompt": prompts[i],
                    "problem": test_data[i]["question"],
                    "answer": test_data[i]["gt"],
                    "solution": test_data[i]["answer"],
                    "model_generation": [trim_output(text)],
                }) + "\n")
            # A batch lands all-or-nothing, and hits the disk before the next
            # generate() call -- which is ~30 min at this decode rate. So a
            # crash, an OOM or a spot reclaim costs one batch, not the run.
            fout.flush()
            os.fsync(fout.fileno())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH",
        help="MATH500 / GSM / LogiQA / MMLU",
    )
    parser.add_argument(
        "--mmlu_subject",
        type=str,
        default="philosophy",
        help="MMLU subject (config) to evaluate when --dataset MMLU.",
    )
    parser.add_argument(
        "--logiqa_english_only",
        action="store_true",
        default=False,
        help="Drop CJK-heavy rows when --dataset LogiQA. LogiQA 2.0 is ~half "
             "untranslated Chinese; this applies the same is_cjk_heavy filter "
             "gen_logiqa_vllm.py uses for the build set. Off by default so "
             "existing mixed-language LogiQA results stay reproducible.",
    )
    parser.add_argument(
        "--logiqa_eval_selection",
        type=str,
        default=None,
        help="JSON with {'items': [{'pool_idx': int}, ...]} naming exactly which "
             "rows to score, overriding --random_sample/--max_examples. Indices "
             "index load_logiqa(english_only=True) order. Use "
             "data/LogiQA/eval_rand42_500_clean.json for the contamination-free set.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--steering",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--steering_vector",
        type=str,
        default=None
    )
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the batches across this many independent processes, one per "
             "GPU. Each writes its own checkpoint; the arm is only scored once "
             "every example is present. The shards never communicate -- there "
             "are no gradients here, so do not reach for torchrun.",
    )
    parser.add_argument(
        "--shard_rank",
        type=int,
        default=0,
        help="Which shard this process is, in [0, num_shards).",
    )
    parser.add_argument(
        "--print_save_dir",
        action="store_true",
        help="Print the derived save_dir and exit, so a launcher can address "
             "the same directory without reimplementing the path logic.",
    )

    args = parser.parse_args()

    if not 0 <= args.shard_rank < args.num_shards:
        parser.error(f"--shard_rank must be in [0, {args.num_shards}), got {args.shard_rank}")

    if args.steering:
        vector_name_split = args.steering_vector.split("/")[-3:]
        vector_name_split[-1] = vector_name_split[-1].split(".")[0]
        name = "_".join(vector_name_split)
        args.save_dir = os.path.join(args.save_dir, name, f"coef_{args.steering_coef}")
    else:
        args.save_dir = os.path.join(args.save_dir, "base")
    
    if args.remove_bos:
        args.save_dir = args.save_dir + "_remove_bos"

    if args.logiqa_eval_selection:
        # Name the leaf after the selection file, so an explicit-selection run can
        # never overwrite a seeded-sample run that happens to share n and seed.
        leaf = os.path.splitext(os.path.basename(args.logiqa_eval_selection))[0]
        args.save_dir = os.path.join(args.save_dir, leaf)
    elif args.random_sample and args.max_examples:
        args.save_dir = os.path.join(args.save_dir, f"rand{args.sample_seed}_{args.max_examples}")
    elif args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")
        
    if args.print_save_dir:
        print(args.save_dir)
        sys.exit(0)

    print(args.save_dir)
    main(args)

    if args.num_shards == 1:
        finalize(args.save_dir, args.dataset)
    else:
        # Scoring waits for every shard; one finishing says nothing about the rest.
        print(f"[shard {args.shard_rank}/{args.num_shards}] generation done. Merge with:\n"
              f"  python scripts/merge_shards.py --save_dir {args.save_dir} --dataset {args.dataset}")


        
