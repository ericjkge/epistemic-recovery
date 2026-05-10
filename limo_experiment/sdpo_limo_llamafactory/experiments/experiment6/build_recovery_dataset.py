#!/usr/bin/env python3
"""Build the experiment 6 SFT dataset by merging two genuine-recovery cohorts.

Sources:
  1. Pass-4 cohort (~77 rows): chains in
     experiment5/outputs/pass4/injection_traces.jsonl where R1 emitted a
     `\\boxed{}` answer that was wrong AND a later round recovered. This is
     the strict "committed-wrong → recovered" cohort from NOTES.md §Pass-4.
  2. Legacy cohort (~126 rows): experiment5/outputs/onpolicy_recovery_dataset.json,
     already in alpaca form, from the earlier 18K-cap pass.

Per the user spec, the two cohorts are concatenated *without* dedupe — we
want enough rows to clear the ~150-problem floor for stable LoRA SFT before
worrying about overlap. Use --dedupe_questions to drop duplicate problems
(keeps first occurrence; pass-4 is listed first).

Output rows match make_limo_dataset.py / make_injection_dataset.py shape:
  {instruction, input="", output, system}
where `output` is `<think>\\n{thinking}\\n</think>\\n\\nThe final answer is \\boxed{N}.`
"""
import argparse
import json
import re
import sys
from pathlib import Path

EXP6_DIR = Path(__file__).resolve().parent
EXP5_DIR = EXP6_DIR.parent / "experiment5"

# Logic mirrored from experiment5/make_injection_dataset.py so dataset rows
# are byte-identical in form to the existing on-policy SFT distribution.
# Inlined (rather than imported) because that module uses PEP 604 syntax
# that breaks on pre-3.10 Python.
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


def iter_pass4_genuine(traces_path: Path):
    """Yield alpaca rows for chains that match the genuine-recovery rule:
       R1 emitted a `\\boxed{}` answer that was wrong, a later round recovered.
    Mirrors NOTES.md §"Recovery dataset (R2-correct cohort)" definitions."""
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for ch in rec.get("chains", []):
                if not ch.get("final_correct"):
                    continue
                rounds = ch.get("rounds", []) or []
                final_round = ch.get("final_round") or len(rounds)
                if final_round < 2 or not rounds:
                    continue  # R1-correct (no recovery to learn from)

                r1 = rounds[0]
                r1_answer = r1.get("extracted_answer")
                if r1_answer is None or str(r1_answer).strip() == "":
                    continue  # pure-truncation: R1 never committed an answer
                if r1.get("correct"):
                    continue  # would mean R1-correct anyway; defensive

                text = ch.get("final_text", "")
                thinking, _ = split_think(text)
                inner_box = extract_last_boxed_inner(text)
                if inner_box is None:
                    continue

                answer_str = str(int(rec["ground_truth"]))
                yield {
                    "instruction": rec["question"],
                    "input": "",
                    "output": build_assistant_output(thinking, answer_str),
                    "system": SYSTEM_PROMPT,
                    "_provenance": "pass4_genuine",
                }


def load_legacy(legacy_path: Path):
    rows = json.loads(legacy_path.read_text())
    for r in rows:
        # Tag provenance without changing the row's training-relevant fields.
        out = dict(r)
        out["_provenance"] = "legacy_recovery_126"
        yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pass4_traces",
        default=str(EXP5_DIR / "outputs" / "pass4" / "injection_traces.jsonl"),
    )
    ap.add_argument(
        "--legacy_dataset",
        default=str(EXP5_DIR / "outputs" / "onpolicy_recovery_dataset.json"),
    )
    ap.add_argument(
        "--output",
        default=str(EXP6_DIR / "outputs" / "genuine_recovery_dataset.json"),
    )
    ap.add_argument(
        "--dedupe_questions",
        action="store_true",
        help="If set, keep only the first row per (normalized) question. Pass-4 "
             "rows are emitted first, so legacy duplicates are the ones dropped.",
    )
    ap.add_argument(
        "--max_chars",
        type=int,
        default=160_000,
        help="Drop rows whose `output` exceeds this many chars. cutoff_len=16K "
             "tokens ≈ 60K chars; 160K is a generous safety margin matching "
             "experiment 5's setting.",
    )
    ap.add_argument(
        "--keep_provenance",
        action="store_true",
        help="Keep the `_provenance` field in the written dataset (default: "
             "stripped before write — LlamaFactory ignores it but it isn't "
             "load-bearing for training).",
    )
    args = ap.parse_args()

    pass4_path = Path(args.pass4_traces)
    legacy_path = Path(args.legacy_dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pass4_path.exists():
        print(f"ERROR: pass-4 traces not found: {pass4_path}", file=sys.stderr)
        return 1
    if not legacy_path.exists():
        print(f"ERROR: legacy dataset not found: {legacy_path}", file=sys.stderr)
        return 1

    rows = []
    rows.extend(iter_pass4_genuine(pass4_path))
    n_pass4 = len(rows)
    rows.extend(load_legacy(legacy_path))
    n_legacy = len(rows) - n_pass4

    # Length filter (post-merge so the cohort counts above reflect raw cohorts)
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

    if not args.keep_provenance:
        for r in rows:
            r.pop("_provenance", None)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"  pass-4 traces           {pass4_path}")
    print(f"  legacy dataset          {legacy_path}")
    print(f"  pass-4 genuine rows     {n_pass4}")
    print(f"  legacy rows             {n_legacy}")
    print(f"  dropped (>{args.max_chars} chars)  {n_dropped_long}")
    if args.dedupe_questions:
        print(f"  dropped (dupe question) {n_dropped_dupe}")
    print(f"  total kept              {len(rows)}")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
