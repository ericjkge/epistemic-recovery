#!/usr/bin/env python3
"""Convert injection_traces.jsonl into a LlamaFactory alpaca-format SFT dataset.

Mirrors make_limo_dataset.py's wrapping convention so the trained LoRA's
inference distribution matches Qwen3 thinking-mode output:

    output = "<think>\\n{combined_thinking}\\n</think>\\n\\nThe final answer is \\boxed{N}."

The combined thinking is the *entire* multi-round trace with its inline
injection markers preserved, minus the final "</think>...\\boxed{N}" suffix
(which we re-emit cleanly). That way the trained model sees retries as part
of natural reasoning, not as a special token sequence.

Filtering: only problems with `final_correct=True`. (Dropping wrong-final
problems is cheap and keeps the SFT signal clean — there are no negative-
example training objectives wired up here.)

Usage:
    python3 make_injection_dataset.py \\
        --traces outputs/injection_traces.jsonl \\
        --output outputs/onpolicy_injection_dataset.json
"""
import argparse
import json
import re
from pathlib import Path

EXP5_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_last_boxed_inner(text: str) -> str | None:
    last = None
    for m in BOXED_RE.finditer(text):
        last = m
    return last.group(1).strip() if last else None


def split_think(text: str) -> tuple[str, str]:
    """(thinking_content, post_think_text). If no </think>, all is thinking and
    post_think_text is ''.

    Handles the empty-`<think></think>` shape (LIMO LoRA) by promoting the
    post-tag reasoning back into the think block — same logic as
    `extract_thinking_span` in eval/analyze_epistemic.py.
    """
    last_close = text.rfind(_THINK_CLOSE)
    last_open = text.rfind(_THINK_OPEN, 0, last_close) if last_close != -1 else text.rfind(_THINK_OPEN)
    if last_close == -1:
        if last_open == -1:
            return text.strip(), ""
        return text[last_open + len(_THINK_OPEN):].strip(), ""

    inner = text[last_open + len(_THINK_OPEN): last_close] if last_open != -1 else text[:last_close]
    post = text[last_close + len(_THINK_CLOSE):].strip()
    if len(inner.strip()) < 10 and len(post) > 100:
        # Empty-think shape: post-tag content is the real reasoning. Drop the
        # final \\boxed{...} answer line from the thinking.
        box_idx = post.rfind("\\boxed{")
        if box_idx > 0:
            line_start = post.rfind("\n", 0, box_idx)
            inner = post[: line_start if line_start != -1 else box_idx].strip()
            post = post[line_start + 1 if line_start != -1 else box_idx:].strip()
        else:
            inner = post
            post = ""
    return inner.strip(), post


def build_assistant_output(thinking: str, answer: str) -> str:
    return f"<think>\n{thinking}\n</think>\n\nThe final answer is \\boxed{{{answer}}}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(EXP5_DIR / "outputs" / "injection_traces.jsonl"))
    ap.add_argument("--output", default=str(EXP5_DIR / "outputs" / "onpolicy_injection_dataset.json"))
    ap.add_argument("--include_round1_correct", action="store_true",
                    help="Include problems solved in round 1 (no injection in trace). "
                         "Default: included — they're real on-policy examples too. Use "
                         "--require_injection to drop them.")
    ap.add_argument("--require_injection", action="store_true",
                    help="Keep ONLY problems that needed >=1 retry. Useful for ablating "
                         "whether the injection signal alone drives the gain.")
    ap.add_argument("--max_chars", type=int, default=160_000,
                    help="Drop traces longer than this many chars in the assembled "
                         "output. The training cutoff_len is 32K tokens ≈ 120K chars; "
                         "160K leaves a margin and avoids silently truncating examples.")
    args = ap.parse_args()

    traces_path = Path(args.traces)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_kept = n_dropped_wrong = n_dropped_no_box = n_dropped_long = 0
    n_round1_only = 0
    records = []

    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)

            if not rec.get("final_correct"):
                n_dropped_wrong += 1
                continue

            n_rounds = rec.get("final_round", len(rec.get("rounds", [])))
            if n_rounds == 1:
                n_round1_only += 1
                if args.require_injection:
                    continue

            text = rec["final_text"]
            thinking, _post = split_think(text)
            inner_box = extract_last_boxed_inner(text)
            if inner_box is None:
                n_dropped_no_box += 1
                continue

            # Use the GROUND TRUTH integer in the boxed answer — `final_correct`
            # already verified the model's extracted answer matches GT, so this
            # just normalizes formatting (e.g. "042" → "42").
            answer_str = str(int(rec["ground_truth"]))

            output = build_assistant_output(thinking, answer_str)
            if len(output) > args.max_chars:
                n_dropped_long += 1
                continue

            records.append({
                "instruction": rec["question"],
                "input": "",
                "output": output,
                "system": SYSTEM_PROMPT,
            })
            n_kept += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  source           {traces_path}")
    print(f"  total traces     {n_total}")
    print(f"  kept             {n_kept}")
    print(f"    round-1 only   {n_round1_only}{' (excluded by --require_injection)' if args.require_injection else ''}")
    print(f"  dropped (wrong)  {n_dropped_wrong}")
    print(f"  dropped (no \\boxed) {n_dropped_no_box}")
    print(f"  dropped (>{args.max_chars} chars) {n_dropped_long}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
