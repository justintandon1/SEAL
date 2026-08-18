import argparse
import glob
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from tqdm import tqdm, trange

import sys
import os
import gc
from code_evaluation import codegen_metrics, load_code_generation_dataset, get_deepseekcode_question_template_answer, extract_code, extract_instance_results
from mbpp_utils import load_mbpp, build_mbpp_prompt, save_mbpp_results
from apps_utils import load_apps, build_apps_prompt, save_apps_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHECKPOINT_GLOB = "predictions.partial*.jsonl"

# Settings a resume has to agree on. Ones already encoded in save_dir (vector
# name, coef) are listed too so the check stands alone; the ones that are not --
# steering_layer, max_tokens, batch_size, model -- are the whole point, since
# nothing else would catch two half-arms merged into one file.
FINGERPRINT_KEYS = [
    "model_name_or_path", "benchmark", "apps_split", "apps_task_file",
    "start", "max_examples", "random_sample", "sample_seed",
    "stratify_by_difficulty", "max_tokens", "batch_size", "use_chat_format",
    "remove_bos", "steering", "steering_vector", "steering_layer",
    "steering_coef", "release",
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
    an example any earlier process finished is still skipped.
    """
    completed = {}
    for path in sorted(glob.glob(os.path.join(save_dir, CHECKPOINT_GLOB))):
        for rec in _read_checkpoint(path):
            completed[rec["idx"]] = rec
    return completed


def check_or_write_fingerprint(args, n_examples):
    """Refuse to resume a checkpoint written under different settings."""
    path = os.path.join(args.save_dir, "run_config.json")
    current = {k: getattr(args, k, None) for k in FINGERPRINT_KEYS}
    current["n_examples"] = n_examples

    if os.path.exists(path):
        with open(path) as fin:
            previous = json.load(fin)
        drift = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
        if drift:
            raise SystemExit(
                f"Refusing to resume: {path} was written under a different config.\n"
                + "\n".join(f"  {k}: checkpoint={pv!r} now={nv!r}" for k, (pv, nv) in sorted(drift.items()))
                + "\nDelete the directory to start clean, or fix the arguments."
            )
        return

    # Atomic, so a concurrent shard never reads a half-written file.
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fout:
        json.dump(current, fout, indent=2, sort_keys=True)
    os.replace(tmp, path)





def main(args):
    random.seed(42)

    print("Loading data...")

    if args.benchmark == "mbpp":
        mbpp_data = load_mbpp(start=args.start, max_examples=args.max_examples)
    elif args.benchmark == "apps":
        apps_data = load_apps(
            split=args.apps_split,
            start=args.start,
            max_examples=args.max_examples,
            random_sample=args.random_sample,
            sample_seed=args.sample_seed,
            task_file=args.apps_task_file,
            stratify_by_difficulty=args.stratify_by_difficulty,
        )
    else:
        benchmark = load_code_generation_dataset(release_version=args.release)

        if args.start:
            benchmark = benchmark[args.start:]

        if args.max_examples and len(benchmark) > args.max_examples:
            benchmark = benchmark[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.benchmark == "mbpp":
        source = mbpp_data
    elif args.benchmark == "apps":
        source = apps_data
    else:
        source = benchmark
    prompts = []
    for example in source:
        if args.benchmark == "mbpp":
            prompt = build_mbpp_prompt(example)
        elif args.benchmark == "apps":
            prompt = build_apps_prompt(example)
        else:
            prompt = get_deepseekcode_question_template_answer(example)
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    if args.shard_rank == 0:
        with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            fout.write(prompts[0])

    check_or_write_fingerprint(args, len(prompts))


    # The batch is the unit of both work and recovery. Batches are left-padded
    # to their longest member, so an example's neighbours are part of its
    # numerics; keeping batch boundaries at fixed global offsets -- across
    # shards and across a resume -- means every example is scored in the same
    # company it would have had in one uninterrupted run. At APPS's
    # batch_size=1 that reduces to per-example recovery.
    batch_starts = list(range(0, len(prompts), args.batch_size))
    my_starts = [s for j, s in enumerate(batch_starts)
                 if j % args.num_shards == args.shard_rank]

    completed = load_completed(args.save_dir)
    todo = [] if args.finalize_only else [
        s for s in my_starts
        if not all(i in completed for i in range(s, min(s + args.batch_size, len(prompts))))
    ]

    tag = "" if args.num_shards == 1 else f"[shard {args.shard_rank}/{args.num_shards}] "
    if args.finalize_only:
        print(f"{tag}finalize only: {len(completed)}/{len(prompts)} predictions on disk")
    else:
        print(f"{tag}{len(my_starts)} batches assigned, "
              f"{len(my_starts) - len(todo)} already checkpointed, {len(todo)} to run")

    checkpoint = checkpoint_path(args.save_dir, args.shard_rank, args.num_shards)
    # Rewrite our own file from its parsed records, so a torn trailing line from
    # a killed run cannot corrupt the lines about to be appended.
    if todo and os.path.exists(checkpoint):
        mine = _read_checkpoint(checkpoint)
        with open(checkpoint, "w") as fout:
            for rec in mine:
                fout.write(json.dumps(rec) + "\n")

    if todo:
        if "qwen" in args.model_name_or_path.lower():
            model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
        else:
            raise ValueError("Model not supported")

        if args.steering:
            steer_vec = torch.load(args.steering_vector, weights_only=True)
            steer_vec = steer_vec.to(model.device)
            model.set_steering_flag(steering_flag=True, steering_layer=args.steering_layer, steer_vec=steer_vec,  steer_coef=args.steering_coef, tokenizer=tokenizer)

        with open(checkpoint, "a") as fout:
            for s in tqdm(todo):
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
                    fout.write(json.dumps({"idx": s + offset, "output": text}) + "\n")
                # A batch lands all-or-nothing and hits the disk before the next
                # generate() call. At APPS's rate that call is many minutes, so a
                # crash, an OOM or a spot reclaim costs one example, not the run.
                fout.flush()
                os.fsync(fout.fileno())

    # Generation is done for this process. Scoring needs every shard's output,
    # and for APPS it executes untrusted test code -- expensive, and not
    # something to run concurrently from several shards. So a sharded run stops
    # here and is scored by one explicit --finalize_only pass.
    completed = load_completed(args.save_dir)
    if args.num_shards > 1 and not args.finalize_only:
        print(f"{tag}generation done. {len(completed)}/{len(prompts)} predictions on disk. "
              f"Once every shard has finished, score with:\n"
              f"  python eval_code_steering.py --finalize_only ... (same arguments, --num_shards 1)")
        return

    missing = [i for i in range(len(prompts)) if i not in completed]
    if missing:
        # A partial arm is worse than no arm: the accuracy still looks plausible.
        raise SystemExit(
            f"Incomplete: {len(missing)}/{len(prompts)} predictions missing "
            f"(first: {missing[:5]}). Re-run the same command -- finished work is skipped."
        )

    outputs = [[completed[i]["output"]] for i in range(len(prompts))]

    if args.benchmark == "mbpp":
        save_mbpp_results(outputs, mbpp_data, args.save_dir, timeout=args.timeout)
        return
    if args.benchmark == "apps":
        save_apps_results(
            outputs,
            apps_data,
            args.save_dir,
            timeout=args.timeout,
            workers=args.eval_workers,
        )
        return

    combined_results = [
        (
            outputs_list,
            [extract_code(output) for output in outputs_list],
        )
        for outputs_list in outputs
    ]

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            benchmark, combined_results
        )
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as f:
        json.dump(save_results, f, indent=4)


    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=12,
        timeout=50,
    )

    print(metrics[0]["pass@1"])

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    with open(os.path.join(args.save_dir, "metrics.jsonl"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(args.save_dir, "code_eval.jsonl"), "w") as f:
        json.dump(save_eval_results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
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
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--release",
        type=str,
        default="release_v1",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="livecodebench",
        choices=["livecodebench", "mbpp", "apps"],
        help="which code benchmark to run",
    )
    parser.add_argument(
        "--apps_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="APPS split used when --benchmark apps.",
    )
    parser.add_argument(
        "--apps_task_file",
        type=str,
        default=None,
        help="optional v_code-SEAL APPS JSONL task list (problem_id order).",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="sample APPS from the full usable split instead of taking a prefix.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="seed used with --random_sample.",
    )
    parser.add_argument(
        "--stratify_by_difficulty",
        action="store_true",
        help="balance an APPS random sample across its three difficulty tiers.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=12,
        help="parallel APPS grading workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="per-test execution timeout in seconds (MBPP/APPS pass@1).",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
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
        help="Split generation across this many processes, one per GPU. Batches "
             "are dealt round-robin at fixed global offsets, so an example keeps "
             "the same batch -- and the same padding -- it would have had on one GPU.",
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
        help="Print the resolved save_dir and exit, so a launcher does not have "
             "to reimplement this script's nesting rules.",
    )
    parser.add_argument(
        "--finalize_only",
        action="store_true",
        help="Skip generation; merge every checkpoint in save_dir and score. "
             "Use once all shards have finished.",
    )


    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error(f"--num_shards must be >= 1, got {args.num_shards}")
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

    if args.benchmark == "apps" and args.random_sample and args.max_examples:
        sampling = "balanced" if args.stratify_by_difficulty else "rand"
        args.save_dir = os.path.join(
            args.save_dir,
            f"{args.apps_split}_{sampling}{args.sample_seed}_{args.max_examples}",
        )
    elif args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")
        
    if args.print_save_dir:
        print(args.save_dir)
        sys.exit(0)

    print(args.save_dir)
    main(args)

        
