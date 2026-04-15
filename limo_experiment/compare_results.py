"""
Compare pre- vs. post-LoRA AIME evaluation summaries.
Prints a side-by-side table of accuracy and epistemic token statistics.

Usage:
  python compare_results.py \
      --pre_dir  ./results/pre_lora \
      --post_dir ./results/post_lora
"""

import argparse
import glob
import json
import os
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_summaries(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "**", "*_summary.json"), recursive=True))


def print_comparison(pre: dict, post: dict, label: str = ""):
    print(f"\n{'='*70}")
    if label:
        print(f"  Run: {label}")
    print(f"  Pre  model : {pre['model']}")
    print(f"  Post model : {post['model']}")
    print(f"  AIME years : {pre.get('aime_years', '?')}  |  n_samples: {pre.get('n_samples', '?')}")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Pre-LoRA':>12} {'Post-LoRA':>12} {'Delta':>10}")
    print(f"{'-'*70}")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def delta(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            d = b - a
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.4f}"
        return "—"

    metrics = [
        ("Accuracy (Pass@1)",         pre["accuracy_pass_at_1"],          post["accuracy_pass_at_1"]),
        ("Correct / Total",            f"{pre['correct_count']}/{pre['n_problems']}",
                                        f"{post['correct_count']}/{post['n_problems']}"),
        ("Avg Response Tokens",        pre["avg_response_tokens"],          post["avg_response_tokens"]),
        ("Epistemic Total",            pre["epistemic"]["total"],           post["epistemic"]["total"]),
        ("Epistemic Avg/Response",     pre["epistemic"]["avg_per_response"], post["epistemic"]["avg_per_response"]),
        ("Epistemic Avg/Problem",      pre["epistemic"]["avg_per_problem"],  post["epistemic"]["avg_per_problem"]),
    ]

    for name, pre_val, post_val in metrics:
        pre_s = fmt(pre_val)
        post_s = fmt(post_val)
        try:
            d = delta(float(pre_val), float(post_val))
        except (TypeError, ValueError):
            d = "—"
        print(f"  {name:<33} {pre_s:>12} {post_s:>12} {d:>10}")

    print(f"\n  Per-word epistemic breakdown:")
    print(f"  {'Word':<16} {'Pre Total':>10} {'Post Total':>11} {'Delta':>8}")
    print(f"  {'-'*48}")
    pre_words = pre["epistemic"]["per_word"]
    post_words = post["epistemic"]["per_word"]
    for w in pre_words:
        pv = pre_words.get(w, 0)
        po = post_words.get(w, 0)
        d = po - pv
        sign = "+" if d >= 0 else ""
        print(f"  {w:<16} {pv:>10} {po:>11} {sign+str(d):>8}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_dir", type=str, required=True,
                        help="Directory containing pre-LoRA *_summary.json files")
    parser.add_argument("--post_dir", type=str, required=True,
                        help="Directory containing post-LoRA *_summary.json files (may contain subdirs per run)")
    args = parser.parse_args()

    pre_summaries = find_summaries(args.pre_dir)
    if not pre_summaries:
        print(f"No *_summary.json found in {args.pre_dir}")
        return

    # Use the first (or only) pre-LoRA summary
    pre = load_summary(pre_summaries[0])
    if len(pre_summaries) > 1:
        print(f"[warn] Multiple pre-LoRA summaries found; using {pre_summaries[0]}")

    post_summaries = find_summaries(args.post_dir)
    if not post_summaries:
        print(f"No *_summary.json found in {args.post_dir}")
        return

    for post_path in post_summaries:
        post = load_summary(post_path)
        label = Path(post_path).parent.name
        print_comparison(pre, post, label=label)

    print("\nDone.")


if __name__ == "__main__":
    main()
