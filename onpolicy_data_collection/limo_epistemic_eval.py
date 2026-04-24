#!/usr/bin/env python3
"""Count epistemic words in LIMO-v2 JSONL eval outputs.

Copied/adapted from eval/check_epistemic_tokens.py so this folder can be used
without reaching into the broader eval utilities.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


EPISTEMIC_WORDS = [
    "wait",
    "hmm",
    "perhaps",
    "maybe",
    "actually",
    "alternatively",
    "seems",
    "might",
    "likely",
    "check",
]


def count_epistemic_words(text: str) -> tuple[dict[str, int], int]:
    text_lower = text.lower()
    counts = {}
    total = 0
    for word in EPISTEMIC_WORDS:
        count = text_lower.count(word)
        counts[word] = count
        total += count
    return counts, total


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Count epistemic words in LIMO-v2 eval JSONL.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--response_key",
        type=str,
        default="generated_responses",
        help="Key for the response list in JSONL, e.g. generated_responses or answer_responses.",
    )
    args = parser.parse_args()

    data = load_jsonl(Path(args.input))
    global_word_counts = defaultdict(int)
    global_total = 0
    total_responses = 0
    per_problem = []

    for i, record in enumerate(data):
        responses = record.get(args.response_key, [])
        prob_counts = defaultdict(int)
        prob_total = 0

        for response in responses:
            counts, total = count_epistemic_words(response)
            for word, count in counts.items():
                prob_counts[word] += count
                global_word_counts[word] += count
            prob_total += total
            global_total += total
            total_responses += 1

        n = len(responses) or 1
        per_problem.append(
            {
                "index": i,
                "num_responses": len(responses),
                "word_totals": dict(prob_counts),
                "total": prob_total,
                "avg": prob_total / n,
            }
        )

    print(f"Problems: {len(data)} | Responses: {total_responses}\n")
    print(f"{'Word':<18} {'Total':<10} {'Avg/Resp':<12} {'Avg/Problem':<12}")
    print("-" * 52)
    for word in EPISTEMIC_WORDS:
        total = global_word_counts[word]
        avg_resp = total / total_responses if total_responses else 0
        avg_problem = total / len(data) if data else 0
        print(f"{word:<18} {total:<10} {avg_resp:<12.3f} {avg_problem:<12.3f}")
    print("-" * 52)
    avg_total_resp = global_total / total_responses if total_responses else 0
    avg_total_problem = global_total / len(data) if data else 0
    print(f"{'TOTAL':<18} {global_total:<10} {avg_total_resp:<12.2f} {avg_total_problem:<12.2f}")


if __name__ == "__main__":
    main()
