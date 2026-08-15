#!/usr/bin/env python
"""Merge the checkpoints from a sharded eval into predictions.jsonl, then score.

Each shard of a multi-GPU run appends to its own predictions.partial.NofM.jsonl.
This stitches them back into the single predictions.jsonl the rest of the
pipeline expects, in global example order, and runs the grader. It refuses to
score a partial run -- an arm missing a batch is worse than no arm, because the
accuracy still looks plausible.

Safe to re-run: it only reads the checkpoints.

    python scripts/merge_shards.py --save_dir <leaf dir> --dataset LogiQA
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from eval_MATH_steering import finalize


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--save_dir",
        required=True,
        help="The leaf results directory holding the checkpoint files. Get it "
             "from eval_MATH_steering.py --print_save_dir.",
    )
    parser.add_argument(
        "--dataset",
        default="MATH500",
        help="MATH500 / GSM / LogiQA / MMLU -- picks the grader.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        raise SystemExit(f"No such directory: {args.save_dir}")

    finalize(args.save_dir, args.dataset)


if __name__ == "__main__":
    main()
