#!/usr/bin/env python3
"""Build seed_problems.jsonl for experiment 5's injection generator.

Default source: GAIR/LIMO-v2 — the same 813 competition-math questions the
off-policy LIMO LoRA (experiments 1, 2, 4) was trained on. Using the same
question set is deliberate: experiment 5 is the on-policy counterpart of
experiments 1–4, and matching the question distribution makes the comparison
clean (off-policy LIMO solutions vs. on-policy traces from the same model
on the same questions).

AIME24 + AIME25 (the held-out eval) are not in LIMO, so no eval contamination.
About 70–80% of LIMO have integer answers; the rest are filtered out by the
integer-grade check (matching the grader in evaluate_aime.py).

Override with --source to point at any HF dataset or a local JSONL — e.g.,
`AI-MO/aimo-validation-aime` (90 pre-2024 AIME problems) for a smaller, fully
disjoint seed set.

Output schema (one record per line):
    {
      "problem_id": "limo_seed_017",
      "question":   "...",
      "answer":     42       # integer
    }

Records with non-integer answers are skipped (we grade with `int()` comparison
in generate_injections.py, mirroring evaluate_aime.py).

Usage:
    python3 prepare_seed_problems.py                              # GAIR/LIMO-v2 (default)
    python3 prepare_seed_problems.py --max 200                    # cap to 200 seeds
    python3 prepare_seed_problems.py --source AI-MO/aimo-validation-aime
    python3 prepare_seed_problems.py --source local:/path/to/my.jsonl
"""
import argparse
import json
import re
from pathlib import Path

EXP5_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EXP5_DIR / "seed_problems.jsonl"


def _coerce_int(val) -> int | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.search(r"-?\d+", s.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def _question_from_row(row: dict) -> str | None:
    for k in ("problem", "question", "Problem", "Question", "query"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _answer_from_row(row: dict) -> int | None:
    for k in ("answer", "Answer", "final_answer", "ground_truth", "solution_answer"):
        ai = _coerce_int(row.get(k))
        if ai is not None:
            return ai
    return None


def load_source(source: str, hf_split: str):
    if source.startswith("local:"):
        path = Path(source[len("local:"):])
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    # Otherwise treat as a HuggingFace dataset name.
    from datasets import load_dataset
    ds = load_dataset(source, split=hf_split)
    yield from ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="GAIR/LIMO-v2",
                    help="HF dataset name or 'local:/path/to.jsonl'. Default GAIR/LIMO-v2 "
                         "matches the question set used by the off-policy LIMO LoRA.")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--max", type=int, default=None,
                    help="Cap the number of accepted records.")
    ap.add_argument("--id_prefix", default="limo_seed",
                    help="Prefix for synthesized problem_ids.")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_kept = n_skipped = 0
    with open(out, "w") as f:
        for i, row in enumerate(load_source(args.source, args.hf_split)):
            q = _question_from_row(row)
            a = _answer_from_row(row)
            if q is None or a is None:
                n_skipped += 1
                continue
            rec = {
                "problem_id": f"{args.id_prefix}_{n_kept:04d}",
                "question": q,
                "answer": a,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1
            if args.max is not None and n_kept >= args.max:
                break

    print(f"  source     {args.source}")
    print(f"  kept       {n_kept}")
    print(f"  skipped    {n_skipped} (no integer answer or no question text)")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
