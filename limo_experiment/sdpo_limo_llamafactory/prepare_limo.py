#!/usr/bin/env python3
"""Reshape GAIR/LIMO into Qwen3 thinking-mode ShareGPT format.

LIMO solutions are flat reasoning prose ending in \\boxed{answer}. Qwen3 thinking mode
expects reasoning inside <think>...</think> with the final answer AFTER the closing tag.
We split each solution at the LAST \\boxed{...} occurrence and route everything before it
into the <think> block (verbatim — epistemic markers must be preserved).
"""
import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset

USER_TEMPLATE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# Match \boxed{ ... } allowing one level of nested braces (common: \boxed{\frac{1}{2}}).
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def find_last_boxed_span(text: str):
    """Return (start_idx, end_idx, inner) of the LAST \\boxed{...} in text, or None."""
    last = None
    for m in BOXED_RE.finditer(text):
        last = m
    if last is None:
        return None
    return last.start(), last.end(), last.group(1).strip()


def split_at_final_boxed(solution: str):
    """Split solution into (thinking_content, answer_str).

    thinking_content is everything before the start of the line/sentence containing the
    final \\boxed{}. answer_str is the inner contents of that final \\boxed{}.

    Returns None if no \\boxed{} or nothing precedes it.
    """
    span = find_last_boxed_span(solution)
    if span is None:
        return None
    start, _, inner = span

    # Walk back from the boxed start to the previous line break, so the entire
    # answer-containing line (e.g. "The final answer is \\boxed{42}.") leaves the think.
    line_start = solution.rfind("\n", 0, start)
    cut = line_start + 1 if line_start != -1 else 0
    thinking_content = solution[:cut].rstrip()
    if not thinking_content:
        return None
    return thinking_content, inner


def build_assistant_response(thinking_content: str, answer: str) -> str:
    return f"<think>\n{thinking_content}\n</think>\n\nThe final answer is \\boxed{{{answer}}}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GAIR/LIMO")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/limo_qwen3_thinking.json")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} raw samples")

    records = []
    skipped_no_boxed = 0
    skipped_no_thinking = 0

    for row in ds:
        question = row.get("question") or row.get("problem")
        solution = row.get("solution") or row.get("response")
        if not question or not solution:
            skipped_no_boxed += 1
            continue

        split = split_at_final_boxed(solution)
        if split is None:
            # Either no \boxed or no thinking before it.
            if find_last_boxed_span(solution) is None:
                skipped_no_boxed += 1
            else:
                skipped_no_thinking += 1
            continue

        thinking_content, boxed_inner = split
        # Prefer dataset's `answer` field if present; fall back to the boxed content.
        answer = row.get("answer")
        if answer is None or str(answer).strip() == "":
            answer = boxed_inner
        answer = str(answer).strip()

        assistant = build_assistant_response(thinking_content, answer)
        records.append(
            {
                "conversations": [
                    {"from": "human", "value": USER_TEMPLATE.format(question=question)},
                    {"from": "gpt", "value": assistant},
                ]
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} samples → {out_path}")
    print(f"  skipped (no \\boxed): {skipped_no_boxed}")
    print(f"  skipped (no thinking before \\boxed): {skipped_no_thinking}")

    # ------------------------------------------------------------------
    # Tokenizer sanity check + length histogram
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer {args.tokenizer_name} for sanity checks...")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    expected_ids = {"<think>": 151667, "</think>": 151668}
    for tag, expected in expected_ids.items():
        ids = tok.encode(tag, add_special_tokens=False)
        ok = len(ids) == 1 and ids[0] == expected
        print(f"  {tag}: ids={ids}  expected single token id={expected}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"    WARNING: {tag} did not tokenize as a single special token. "
                  "Check that you are using the Qwen3 tokenizer.")

    # Length distribution over the assistant turn (most relevant for cutoff_len).
    lengths = [
        len(tok.encode(r["conversations"][1]["value"], add_special_tokens=False))
        for r in records
    ]
    if lengths:
        lengths.sort()
        n = len(lengths)
        p50 = lengths[n // 2]
        p95 = lengths[int(n * 0.95)]
        print(f"\nAssistant-turn token lengths: p50={p50}  p95={p95}  max={lengths[-1]}")


if __name__ == "__main__":
    main()
