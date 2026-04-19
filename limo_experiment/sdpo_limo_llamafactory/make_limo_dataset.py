#!/usr/bin/env python3
"""Convert GAIR/LIMO-v2 to LlamaFactory alpaca format for SDPO Qwen3-8B LoRA training.

Mirrors strategic-information-allocation-llm-reasoning/make_limo_dataset.py exactly:
raw solution text as output, no <think> wrapping, no answer appending.
"""
import argparse, json, os
from datasets import load_dataset

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="limo_v2_sdpo.json")
    parser.add_argument("--dataset", default="GAIR/LIMO-v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    records = [
        {
            "instruction": row["question"],
            "input": "",
            "output": row["solution"],
            "system": SYSTEM_PROMPT,
        }
        for row in ds
    ]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} examples → {args.output}")


if __name__ == "__main__":
    main()
