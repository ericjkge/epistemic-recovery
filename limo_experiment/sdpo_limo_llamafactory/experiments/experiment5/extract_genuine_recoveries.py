#!/usr/bin/env python3
"""Extract genuine-recovery chains from an injection_traces.jsonl into an
alpaca-format SFT dataset (instruction / input / output / system).

"Genuine recovery" rule (mirrors exp6's iter_pass4_genuine):
  - chain final_correct=True
  - chain has >= 2 rounds (otherwise R1-correct → nothing to learn)
  - R1 emitted a non-empty `\\boxed{...}` (so R1 committed a wrong answer
    rather than being a pure-truncation case — those aren't the recovery
    signal we want)
  - R1 was wrong (defensive — `final_correct` already implies a later round
    flipped, but we double-check)

Built as a standalone utility (vs the exp6 script) because exp6's version
requires concatenation with a LIMO legacy cohort that doesn't apply to the
DAPO pass. Once a second DAPO pass lands, use --traces multiple times to
merge cohorts before writing.

Output rows are byte-identical in form to make_injection_dataset.py /
build_recovery_dataset.py:
  {"instruction": question,
   "input": "",
   "output": "<think>\\n{thinking}\\n</think>\\n\\nThe final answer is \\boxed{N}.",
   "system": "Please reason step by step, and put your final answer within \\boxed{}."}

Usage:
    python3 extract_genuine_recoveries.py \\
        --traces outputs/dapo_pass1/injection_traces.jsonl \\
        --output outputs/dapo_pass1/genuine_recovery_dataset.json
"""
import argparse
import json
import re
from pathlib import Path

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_last_boxed_inner(text):
    last = None
    for m in BOXED_RE.finditer(text):
        last = m
    return last.group(1).strip() if last else None


def split_think(text):
    """Return (thinking_content, post_think_text). Handles the empty-`<think></think>`
    shape (LIMO LoRA) by promoting post-tag reasoning back into the think block."""
    last_close = text.rfind(_THINK_CLOSE)
    last_open = text.rfind(_THINK_OPEN, 0, last_close) if last_close != -1 else text.rfind(_THINK_OPEN)
    if last_close == -1:
        if last_open == -1:
            return text.strip(), ""
        return text[last_open + len(_THINK_OPEN):].strip(), ""

    inner = text[last_open + len(_THINK_OPEN): last_close] if last_open != -1 else text[:last_close]
    post = text[last_close + len(_THINK_CLOSE):].strip()
    if len(inner.strip()) < 10 and len(post) > 100:
        box_idx = post.rfind("\\boxed{")
        if box_idx > 0:
            line_start = post.rfind("\n", 0, box_idx)
            inner = post[: line_start if line_start != -1 else box_idx].strip()
            post = post[line_start + 1 if line_start != -1 else box_idx:].strip()
        else:
            inner = post
            post = ""
    return inner.strip(), post


def build_assistant_output(thinking, answer):
    return "<think>\n" + thinking + "\n</think>\n\nThe final answer is \\boxed{" + answer + "}."


# Stricter filter than `final_correct=True`: require that the LAST \boxed{}
# inside the THINKING section integer-matches GT. The base grader takes
# whichever \boxed{} appears last anywhere in the trace, which lets through
# chains whose thinking concludes wrong but emits a stray \boxed{GT} earlier
# or as a post-</think> aside (we discard the post-</think> section when
# building the SFT row, so training on those teaches "think wrong, announce
# right" — the opposite of what we want). Auditing the pass-1 cohort showed
# ~13% of final_correct chains had this pattern.
_INT_DIGITS = re.compile(r"-?\d+")


def _coerce_int(s) -> int | None:
    if s is None:
        return None
    m = _INT_DIGITS.search(str(s).replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def iter_genuine(traces_path: Path, provenance_tag: str):
    """Yield SFT rows for chains where R1 committed wrong, a later round
    recovered, AND the thinking section's last \\boxed{} integer-matches GT
    (so the trace we train on actually concludes correctly)."""
    n_dropped_thinking_mismatch = 0
    n_dropped_thinking_no_box = 0
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt_int = _coerce_int(rec["ground_truth"])
            for ch in rec.get("chains", []):
                if not ch.get("final_correct"):
                    continue
                rounds = ch.get("rounds") or []
                final_round = ch.get("final_round") or len(rounds)
                if final_round < 2 or not rounds:
                    continue
                r1 = rounds[0]
                r1_ans = r1.get("extracted_answer")
                if r1_ans is None or str(r1_ans).strip() == "":
                    continue   # pure-truncation finish, not a genuine recovery
                if r1.get("correct"):
                    continue   # defensive
                text = ch.get("final_text", "")
                thinking, _ = split_think(text)
                if extract_last_boxed_inner(text) is None:
                    continue

                thinking_last = extract_last_boxed_inner(thinking)
                if thinking_last is None:
                    n_dropped_thinking_no_box += 1
                    continue
                if _coerce_int(thinking_last) != gt_int:
                    n_dropped_thinking_mismatch += 1
                    continue

                answer_str = str(gt_int)
                yield {
                    "instruction": rec["question"],
                    "input": "",
                    "output": build_assistant_output(thinking, answer_str),
                    "system": SYSTEM_PROMPT,
                    "_provenance": provenance_tag,
                    "_problem_id": rec.get("problem_id"),
                    "_chain_idx": ch.get("chain_idx"),
                    "_r1_extracted_answer": r1_ans,
                    "_thinking_last_boxed": thinking_last,
                    "_ground_truth": rec["ground_truth"],
                }
    if n_dropped_thinking_no_box or n_dropped_thinking_mismatch:
        print(f"    dropped (thinking has no \\boxed):       {n_dropped_thinking_no_box}")
        print(f"    dropped (thinking-last \\boxed != GT):   {n_dropped_thinking_mismatch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", action="append", required=True,
                    help="injection_traces.jsonl to extract from. Pass multiple "
                         "times to merge cohorts (e.g. dapo_pass1 + dapo_pass2).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--strip_metadata", action="store_true",
                    help="Drop _provenance/_problem_id/_chain_idx/_r1_extracted_answer/"
                         "_ground_truth fields. Useful for SFT-ready output that "
                         "LlamaFactory ingests without complaint.")
    ap.add_argument("--max_chars", type=int, default=160_000,
                    help="Drop rows whose `output` exceeds this many chars.")
    ap.add_argument("--dedupe_questions", action="store_true",
                    help="Keep only the first row per normalized question across "
                         "all --traces inputs.")
    args = ap.parse_args()

    rows = []
    for traces in args.traces:
        tag = Path(traces).parent.name or Path(traces).stem
        n_before = len(rows)
        rows.extend(iter_genuine(Path(traces), provenance_tag=tag))
        print(f"  {traces}: +{len(rows) - n_before} genuine recoveries")

    pre_len = len(rows)
    rows = [r for r in rows if len(r["output"]) <= args.max_chars]
    n_dropped_long = pre_len - len(rows)

    n_dropped_dupe = 0
    if args.dedupe_questions:
        seen = set()
        out_rows = []
        for r in rows:
            key = re.sub(r"\s+", " ", r["instruction"]).strip().lower()
            if key in seen:
                n_dropped_dupe += 1
                continue
            seen.add(key)
            out_rows.append(r)
        rows = out_rows

    if args.strip_metadata:
        for r in rows:
            for k in ("_provenance", "_problem_id", "_chain_idx",
                      "_r1_extracted_answer", "_ground_truth"):
                r.pop(k, None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"  rows kept           {len(rows)}")
    print(f"  dropped (>{args.max_chars} chars)  {n_dropped_long}")
    if args.dedupe_questions:
        print(f"  dropped (dupe q)    {n_dropped_dupe}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
