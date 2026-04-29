#!/usr/bin/env python3
"""Convert GAIR/LIMO-v2 to LlamaFactory alpaca format for SDPO Qwen3-8B LoRA training.

Wraps each LIMO solution in <think>...</think> so the assistant output matches the
Qwen3 thinking-mode format the chat template enforces at inference time. Without
wrapping, the LoRA learns to immediately close the auto-opened <think> tag and dump
raw reasoning after it (verified empirically in experiments/experiment2 — 100% empty
<think></think> in the LoRA's AIME outputs).

Wrapping logic: split each LIMO solution at the LAST `\\boxed{...}` occurrence. The
text before that line becomes thinking content; the boxed inner becomes the final
answer. The assistant output is then:

    <think>\\n{thinking_content}\\n</think>\\n\\nThe final answer is \\boxed{{{answer}}}.

Examples without a parseable \\boxed{} or with no reasoning before it are skipped
(small fraction in practice).
"""
import argparse
import json
import os

from datasets import load_dataset

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

BOXED_OPEN = "\\boxed{"


def find_last_boxed_span(text: str):
    """Return (start_idx, end_idx, inner_str) of the LAST \\boxed{...}, or None.

    Uses brace-balanced walking instead of a regex so deeply nested expressions
    (e.g. \\boxed{\\dfrac{5\\sqrt{11}}{4}}) parse correctly. If the \\boxed{...
    is truncated mid-expression (LIMO-v2 has ~5% of solutions cut off this way),
    end_idx = len(text) and inner is the partial content; the caller decides
    how to use the (likely-also-truncated) inner — usually it's overridden by
    the row's `answer` field anyway.
    """
    start = text.rfind(BOXED_OPEN)
    if start == -1:
        return None
    inner_start = start + len(BOXED_OPEN)
    depth = 1
    i = inner_start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1, text[inner_start:i].strip()
        i += 1
    # Unclosed (truncated) — return what we have; caller will use row['answer'].
    # end_idx == len(text) signals truncation to the caller.
    return start, len(text), text[inner_start:].strip()


def split_at_final_boxed(solution: str):
    """Split solution into (thinking_content, partial_answer, was_truncated). None if no \\boxed{ at all."""
    span = find_last_boxed_span(solution)
    if span is None:
        return None
    start, end, inner = span
    was_truncated = end == len(solution) and not solution.endswith("}")
    # Walk back to the previous newline so the entire answer-containing line
    # leaves the think block (e.g. "The final answer is \\boxed{42}.").
    line_start = solution.rfind("\n", 0, start)
    cut = line_start + 1 if line_start != -1 else 0
    thinking_content = solution[:cut].rstrip()
    if not thinking_content:
        return None
    return thinking_content, inner, was_truncated


def build_assistant_output(thinking_content: str, answer: str) -> str:
    return f"<think>\n{thinking_content}\n</think>\n\nThe final answer is \\boxed{{{answer}}}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="limo_v2_sdpo.json")
    parser.add_argument("--dataset", default="GAIR/LIMO-v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--no_wrap",
        action="store_true",
        help="Emit raw solutions without <think>...</think> wrapping (legacy behavior).",
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    records = []
    skipped_no_boxed = 0
    skipped_no_thinking = 0
    recovered_truncated = 0

    for row in ds:
        question = row["question"]
        solution = row["solution"]

        if args.no_wrap:
            output = solution
        else:
            split = split_at_final_boxed(solution)
            if split is None:
                if find_last_boxed_span(solution) is None:
                    skipped_no_boxed += 1
                else:
                    skipped_no_thinking += 1
                continue
            thinking, boxed_inner, was_truncated = split
            # Prefer the dataset's `answer` field. ~5% of LIMO-v2 solutions are
            # truncated mid-\\boxed{... and `boxed_inner` is partial — but the
            # `answer` field is reliably populated, so we recover those rows.
            answer = row.get("answer")
            if answer is None or str(answer).strip() == "":
                answer = boxed_inner
            elif was_truncated:
                recovered_truncated += 1
            output = build_assistant_output(thinking, str(answer).strip())

        records.append({
            "instruction": question,
            "input": "",
            "output": output,
            "system": SYSTEM_PROMPT,
        })

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} examples → {args.output}")
    if not args.no_wrap:
        print(f"  recovered from truncated solution (used row['answer']): {recovered_truncated}")
        print(f"  skipped (no \\boxed at all): {skipped_no_boxed}")
        print(f"  skipped (no reasoning before \\boxed): {skipped_no_thinking}")


if __name__ == "__main__":
    main()
