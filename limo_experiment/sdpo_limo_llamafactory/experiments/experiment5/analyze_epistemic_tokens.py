#!/usr/bin/env python3
"""Per-token epistemic verbalization counts and densities for an injection-traces run.

For each chain in `injection_traces.jsonl`:
  - R1 thinking  = text before the first injection-phrase marker, then the
                   standard <think>...</think> span.
  - FULL thinking = standard <think>...</think> span over the entire trace
                    (R1 reasoning + injection + R2 continuation).
  - R2 thinking   = FULL minus R1 (the new content added after R1, including
                    the injection phrase itself; the trained LoRA will see
                    the injection as part of its own distribution).

Reports, per cohort:
  - count of each of the 10 Kim et al. epistemic tokens in R1, R2, FULL
  - density = epistemic tokens / total words × 1000  (per 1K words)
    pooled across the cohort (Σ epistemic / Σ words), not per-trace averaged

Densities are pooled rather than per-trace averaged because per-trace averaging
weights every chain equally regardless of length, which can hide a real
density shift on the (smaller) cohort of long traces. Pooled density
answers "what fraction of words across the cohort were epistemic."

The injection phrase is read from `<traces_dir>/stats.json` if present,
falling back to `generate_injections.INJECTION_PHRASE` for legacy traces.

Usage:
    python3 analyze_epistemic_tokens.py
    python3 analyze_epistemic_tokens.py --traces outputs/pass1/injection_traces.jsonl
    python3 analyze_epistemic_tokens.py --bucket R2-correct
"""
import argparse
import json
import sys
from pathlib import Path

EXP5_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP5_DIR.parent.parent  # sdpo_limo_llamafactory/
sys.path.insert(0, str(PARENT_DIR / "eval"))
sys.path.insert(0, str(EXP5_DIR))
from analyze_epistemic import (  # type: ignore  # noqa: E402
    EPISTEMIC_TOKENS,
    TOKEN_REGEXES,
    extract_thinking_span,
)
from generate_injections import INJECTION_PHRASE as DEFAULT_INJECTION_PHRASE  # type: ignore  # noqa: E402


def _marker_from_phrase(phrase: str) -> str:
    """Leading 6 words of the phrase — distinctive enough not to collide with
    native model output, short enough to survive minor formatting drift."""
    return " ".join(phrase.strip().split()[:6])


def resolve_injection_phrase(traces_path: Path) -> str:
    """Look up the phrase that was used to produce these traces.

    Order of preference:
      1. `<traces_dir>/stats.json` → `injection_phrase`
      2. The default imported from generate_injections.
    """
    stats_path = traces_path.parent / "stats.json"
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            phrase = stats.get("injection_phrase")
            if isinstance(phrase, str) and phrase.strip():
                return phrase
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_INJECTION_PHRASE


def split_r1_text(final_text: str, marker: str) -> str:
    """Return only the R1 portion of a multi-round trace (text before the
    first injection-phrase marker). For R1-correct chains with no injection
    in the trace, returns final_text unchanged."""
    idx = final_text.find(marker)
    return final_text[:idx].rstrip() if idx != -1 else final_text


def count_per_token(text: str) -> dict[str, int]:
    return {t: len(rgx.findall(text)) for t, rgx in TOKEN_REGEXES.items()}


def words(text: str) -> int:
    return len(text.split())


def density(num: int, den: int) -> float:
    return 1000.0 * num / den if den else 0.0


def collect_rows(traces_path: Path, marker: str) -> list[dict]:
    """One row per chain. Each row carries R1/R2/FULL token counts and word
    counts so cohorts can be filtered downstream without re-parsing."""
    rows = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            chains = rec.get("chains") or [{
                "final_correct": rec.get("final_correct", False),
                "final_text": rec.get("final_text", ""),
                "rounds": rec.get("rounds", []),
            }]
            for ch in chains:
                final_text = ch.get("final_text", "")
                final_correct = ch.get("final_correct", False)
                rounds = ch.get("rounds", []) or []
                final_round = rounds[-1].get("round", 0) if rounds else 0

                if final_correct:
                    bucket = f"R{final_round}-correct"
                else:
                    bucket = "never-correct"

                r1_text = split_r1_text(final_text, marker)
                r1_thinking, _ = extract_thinking_span(r1_text)
                full_thinking, _ = extract_thinking_span(final_text)

                r1_cnt = count_per_token(r1_thinking)
                full_cnt = count_per_token(full_thinking)
                r2_cnt = {t: full_cnt[t] - r1_cnt[t] for t in EPISTEMIC_TOKENS}
                r1_w = words(r1_thinking)
                full_w = words(full_thinking)
                r2_w = full_w - r1_w

                rows.append({
                    "bucket": bucket,
                    "r1_cnt": r1_cnt, "r1_w": r1_w,
                    "full_cnt": full_cnt, "full_w": full_w,
                    "r2_cnt": r2_cnt, "r2_w": r2_w,
                })
    return rows


def report(label: str, rows: list[dict]) -> None:
    n = len(rows)
    print(f"=== {label} (n={n}) ===")
    if n == 0:
        print("  (no chains in this cohort)\n")
        return

    sum_r1 = {t: sum(r["r1_cnt"][t] for r in rows) for t in EPISTEMIC_TOKENS}
    sum_full = {t: sum(r["full_cnt"][t] for r in rows) for t in EPISTEMIC_TOKENS}
    sum_r2 = {t: sum(r["r2_cnt"][t] for r in rows) for t in EPISTEMIC_TOKENS}
    sum_r1_w = sum(r["r1_w"] for r in rows)
    sum_full_w = sum(r["full_w"] for r in rows)
    sum_r2_w = sum(r["r2_w"] for r in rows)

    total_r1 = sum(sum_r1.values())
    total_r2 = sum(sum_r2.values())
    total_full = sum(sum_full.values())

    print(f"  {'token':<14s}  {'R1 cnt':>8s}  {'R2 cnt':>8s}  {'FULL cnt':>9s}     "
          f"{'R1/1Kw':>7s}  {'R2/1Kw':>7s}  {'FULL/1Kw':>8s}")
    print("  " + "─" * 90)
    for tok in EPISTEMIC_TOKENS:
        r1c, r2c, fc = sum_r1[tok], sum_r2[tok], sum_full[tok]
        r1d = density(r1c, sum_r1_w)
        r2d = density(r2c, sum_r2_w)
        fd = density(fc, sum_full_w)
        print(f"  {tok:<14s}  {r1c:>8d}  {r2c:>8d}  {fc:>9d}     "
              f"{r1d:>7.3f}  {r2d:>7.3f}  {fd:>8.3f}")
    print("  " + "─" * 90)
    r1d = density(total_r1, sum_r1_w)
    r2d = density(total_r2, sum_r2_w)
    fd = density(total_full, sum_full_w)
    print(f"  {'TOTAL':<14s}  {total_r1:>8d}  {total_r2:>8d}  {total_full:>9d}     "
          f"{r1d:>7.3f}  {r2d:>7.3f}  {fd:>8.3f}")
    print(f"  {'words':<14s}  {sum_r1_w:>8d}  {sum_r2_w:>8d}  {sum_full_w:>9d}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces",
                    default=str(EXP5_DIR / "outputs" / "pass1" / "injection_traces.jsonl"),
                    help="Path to injection_traces.jsonl. Defaults to outputs/pass1/...")
    ap.add_argument("--bucket",
                    choices=("all-cohorts", "ALL", "R1-correct", "R2-correct",
                             "R3-correct", "never-correct"),
                    default="all-cohorts",
                    help="Filter to a single cohort. Default 'all-cohorts' prints "
                         "one block per cohort plus an ALL block.")
    ap.add_argument("--injection_phrase", default=None,
                    help="Override the phrase used to split R1/R2. Useful for legacy "
                         "trace files whose stats.json predates the injection_phrase "
                         "field — without this, the script falls back to "
                         "generate_injections.INJECTION_PHRASE which may not match.")
    args = ap.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found", file=sys.stderr)
        return 1

    if args.injection_phrase is not None:
        phrase = args.injection_phrase
        phrase_source = "CLI override"
    else:
        phrase = resolve_injection_phrase(traces_path)
        phrase_source = "stats.json or default"
    marker = _marker_from_phrase(phrase)
    print(f"  traces:           {traces_path}")
    print(f"  injection_phrase: {phrase!r}  ({phrase_source})")
    print(f"  R1/full marker:   {marker!r}")
    print()

    rows = collect_rows(traces_path, marker)
    if not rows:
        print("ERROR: no chains found", file=sys.stderr)
        return 1

    print("  Density = epistemic tokens / total words × 1000   (pooled across cohort)")
    print("  R2 = FULL − R1 (the new content added after R1, includes the injection phrase)")
    print()

    if args.bucket == "all-cohorts":
        report("ALL", rows)
        for bucket in ("R1-correct", "R2-correct", "R3-correct", "never-correct"):
            subset = [r for r in rows if r["bucket"] == bucket]
            if subset:
                report(bucket, subset)
    elif args.bucket == "ALL":
        report("ALL", rows)
    else:
        subset = [r for r in rows if r["bucket"] == args.bucket]
        report(args.bucket, subset)

    return 0


if __name__ == "__main__":
    sys.exit(main())
