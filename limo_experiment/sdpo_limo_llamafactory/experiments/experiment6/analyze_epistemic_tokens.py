#!/usr/bin/env python3
"""Epistemic-token counts and pooled densities for the experiment 6 SFT dataset.

Operates on the merged genuine_recovery_dataset.json. Each row's `output`
field is the assistant turn — extract the <think>...</think> span, count the
10 Kim-et-al. epistemic tokens, and report pooled density (Σ epistemic / Σ
words × 1000).

If --by_provenance is set and the dataset still carries the `_provenance`
field (build with `--keep_provenance`), prints one block per provenance
bucket so you can compare pass-4 vs legacy density before training.

Usage:
    python3 analyze_epistemic_tokens.py
    python3 analyze_epistemic_tokens.py --dataset outputs/genuine_recovery_dataset.json --by_provenance
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP6_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP6_DIR.parent.parent  # sdpo_limo_llamafactory/
sys.path.insert(0, str(PARENT_DIR / "eval"))
from analyze_epistemic import (  # type: ignore  # noqa: E402
    EPISTEMIC_TOKENS,
    TOKEN_REGEXES,
    extract_thinking_span,
)


def count_per_token(text: str) -> dict[str, int]:
    return {t: len(rgx.findall(text)) for t, rgx in TOKEN_REGEXES.items()}


def report(label: str, rows: list[dict]) -> None:
    n = len(rows)
    print(f"=== {label} (n={n}) ===")
    if n == 0:
        print("  (no rows)\n")
        return

    sum_cnt = {t: 0 for t in EPISTEMIC_TOKENS}
    sum_words = 0
    for r in rows:
        thinking, _ = extract_thinking_span(r["output"])
        sum_words += len(thinking.split())
        for t, c in count_per_token(thinking).items():
            sum_cnt[t] += c

    total = sum(sum_cnt.values())
    density = lambda num: 1000.0 * num / sum_words if sum_words else 0.0

    print(f"  {'token':<14s}  {'count':>8s}  {'/1Kw':>7s}")
    print("  " + "─" * 35)
    for tok in EPISTEMIC_TOKENS:
        print(f"  {tok:<14s}  {sum_cnt[tok]:>8d}  {density(sum_cnt[tok]):>7.3f}")
    print("  " + "─" * 35)
    print(f"  {'TOTAL':<14s}  {total:>8d}  {density(total):>7.3f}")
    print(f"  {'words':<14s}  {sum_words:>8d}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=str(EXP6_DIR / "outputs" / "genuine_recovery_dataset.json"),
    )
    ap.add_argument(
        "--by_provenance",
        action="store_true",
        help="Print one block per `_provenance` value (requires the dataset "
             "to have been built with --keep_provenance).",
    )
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1

    rows = json.loads(path.read_text())
    print(f"  dataset: {path}  ({len(rows)} rows)")
    print()

    report("ALL", rows)

    if args.by_provenance:
        if not any("_provenance" in r for r in rows):
            print("  (no `_provenance` field — rebuild with --keep_provenance)\n")
            return 0
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r.get("_provenance", "unknown")].append(r)
        for prov in sorted(groups):
            report(prov, groups[prov])

    return 0


if __name__ == "__main__":
    sys.exit(main())
