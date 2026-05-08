#!/usr/bin/env python3
"""Compare epistemic-token verbalization in R1-only vs full multi-round traces.

For each problem in injection_traces.jsonl:
  - R1 thinking = the text before the FIRST injection phrase, then take the
    standard <think>...</think> span (or, for R1-correct problems with no
    injections, the full final_text's think span).
  - Full thinking = the standard <think>...</think> span over all rounds,
    including the injected "Wait, the answer is wrong. Let's think again."
    text. The injection counts AS epistemic verbalization on purpose: the
    trained LoRA learns to emit those phrases as part of its own reasoning,
    so they're part of the verbalization signal we want to measure.

Epistemic tokens are the Kim et al. 10-token set (wait, hmm, perhaps, maybe,
actually, alternatively, seems, might, likely, check), reused from
eval/analyze_epistemic.py for grader consistency.

Reports per-response counts (raw) and density per 1,000 thinking words, both
overall and split by final_round (which round the problem was solved on, or
'never').

Usage:
    python3 analyze_injection_epistemic.py
    python3 analyze_injection_epistemic.py --traces outputs/injection_traces.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

EXP5_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP5_DIR.parent.parent  # sdpo_limo_llamafactory/
sys.path.insert(0, str(PARENT_DIR / "eval"))
from analyze_epistemic import (  # type: ignore  # noqa: E402
    EPISTEMIC_TOKENS,
    TOKEN_REGEXES,
    extract_thinking_span,
)


def split_r1_text(final_text: str) -> str:
    """Return only the R1 portion of a multi-round trace.

    The trace builder constructs R2's input as:
        reopen_thinking(R1_text) + INJECTION_PHRASE
    where reopen_thinking strips R1's </think>+answer suffix. So the R1
    portion ends right before the first INJECTION_PHRASE in final_text.

    For R1-correct problems (no injection), final_text == R1 text, so we
    return the full input.
    """
    idx = final_text.find("Wait, the answer is wrong")
    if idx == -1:
        return final_text
    return final_text[:idx].rstrip()


def count_tokens(text: str) -> dict[str, int]:
    return {t: len(rgx.findall(text)) for t, rgx in TOKEN_REGEXES.items()}


def thinking_word_count(text: str) -> int:
    return len(text.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(EXP5_DIR / "outputs" / "injection_traces.jsonl"))
    ap.add_argument("--per_problem_csv", default=None,
                    help="Optional path to dump per-problem counts as CSV.")
    args = ap.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found", file=sys.stderr)
        return 1

    rows = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            final_text = rec["final_text"]
            n_rounds = len(rec["rounds"])
            final_correct = rec["final_correct"]
            if final_correct:
                bucket = f"R{rec['final_round']}-correct"
            else:
                bucket = "never-correct"

            r1_text = split_r1_text(final_text)
            r1_thinking, _ = extract_thinking_span(r1_text)
            full_thinking, _ = extract_thinking_span(final_text)

            r1_counts = count_tokens(r1_thinking)
            full_counts = count_tokens(full_thinking)
            r1_total = sum(r1_counts.values())
            full_total = sum(full_counts.values())
            r1_words = thinking_word_count(r1_thinking)
            full_words = thinking_word_count(full_thinking)

            rows.append({
                "problem_id": rec["problem_id"],
                "n_rounds": n_rounds,
                "final_correct": final_correct,
                "bucket": bucket,
                "r1_words": r1_words,
                "full_words": full_words,
                "r1_total": r1_total,
                "full_total": full_total,
                "r1_per_token": r1_counts,
                "full_per_token": full_counts,
            })

    if not rows:
        print("ERROR: no traces found", file=sys.stderr)
        return 1

    def density(total: int, words: int) -> float:
        return 1000.0 * total / words if words > 0 else 0.0

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def fmt_block(label: str, subset: list[dict]) -> str:
        n = len(subset)
        if n == 0:
            return f"  {label:<20s}  n=0"
        r1_total = avg([r["r1_total"] for r in subset])
        full_total = avg([r["full_total"] for r in subset])
        r1_density = avg([density(r["r1_total"], r["r1_words"]) for r in subset])
        full_density = avg([density(r["full_total"], r["full_words"]) for r in subset])
        r1_words = avg([r["r1_words"] for r in subset])
        full_words = avg([r["full_words"] for r in subset])
        delta_count = full_total - r1_total
        delta_density = full_density - r1_density
        return (
            f"  {label:<20s}  n={n:>3}  "
            f"R1: {r1_total:6.2f} tokens / {r1_words:6.0f} words "
            f"({r1_density:5.2f}/1K)  "
            f"FULL: {full_total:6.2f} tokens / {full_words:6.0f} words "
            f"({full_density:5.2f}/1K)  "
            f"Δcount={delta_count:+6.2f}  Δdensity={delta_density:+5.2f}"
        )

    print(f"Loaded {len(rows)} traces from {traces_path}")
    print()
    print("Epistemic-token verbalization, R1-only thinking vs FULL multi-round thinking")
    print("(injection-phrase text included in counts — it IS part of the verbalization signal)")
    print()
    print("─" * 130)
    print(fmt_block("ALL PROBLEMS", rows))
    print("─" * 130)
    for bucket in ("R1-correct", "R2-correct", "R3-correct", "never-correct"):
        subset = [r for r in rows if r["bucket"] == bucket]
        print(fmt_block(bucket, subset))
    print("─" * 130)
    print()

    recovered = [r for r in rows if r["bucket"] in ("R2-correct", "R3-correct")]
    print(f"Per-token breakdown for INJECTION-RECOVERED problems (n={len(recovered)}):")
    print(f"  {'token':<14s}  {'R1 avg':>8s}  {'FULL avg':>8s}  {'Δ':>8s}")
    for tok in EPISTEMIC_TOKENS:
        r1_avg = avg([r["r1_per_token"][tok] for r in recovered])
        full_avg = avg([r["full_per_token"][tok] for r in recovered])
        print(f"  {tok:<14s}  {r1_avg:8.2f}  {full_avg:8.2f}  {full_avg - r1_avg:+8.2f}")
    print()

    if args.per_problem_csv:
        import csv
        out = Path(args.per_problem_csv)
        with open(out, "w") as f:
            w = csv.writer(f)
            w.writerow([
                "problem_id", "bucket", "n_rounds", "final_correct",
                "r1_words", "r1_total", "r1_density",
                "full_words", "full_total", "full_density",
            ])
            for r in rows:
                w.writerow([
                    r["problem_id"], r["bucket"], r["n_rounds"], r["final_correct"],
                    r["r1_words"], r["r1_total"], density(r["r1_total"], r["r1_words"]),
                    r["full_words"], r["full_total"], density(r["full_total"], r["full_words"]),
                ])
        print(f"  per-problem CSV → {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
