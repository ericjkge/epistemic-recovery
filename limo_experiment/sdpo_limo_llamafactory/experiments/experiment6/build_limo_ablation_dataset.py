#!/usr/bin/env python3
"""Build the LIMO off-policy ablation dataset for experiment 6.

Samples N rows uniformly at random (with a fixed seed) from the parent's
`limo_v2_sdpo.json` (LlamaFactory alpaca format, ~800 LIMO-v2 rows). Output
matches the alpaca schema used everywhere else in this repo.

Goal: isolate the on-policy / off-policy axis. Holding row count, hparams,
prompt wrapper, and answer format constant, swap the *trace source* — model
recovery traces (exp6 main) vs. LIMO human-written solutions (this script).

Default N=200 to match the user spec for the ablation. The genuine-recovery
cohort lands at ~204 rows; if you want exact-match counts, pass
`--match_recovery_count` and the script will read
`outputs/genuine_recovery_dataset.json` and use its row count.
"""
import argparse
import json
import random
import sys
from pathlib import Path

EXP6_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP6_DIR.parent.parent  # sdpo_limo_llamafactory/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limo_dataset",
        default=str(PARENT_DIR / "limo_v2_sdpo.json"),
        help="Source LIMO alpaca dataset (run parent train.sh once if missing).",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of rows to sample. Default 200 matches the ablation spec.",
    )
    ap.add_argument(
        "--match_recovery_count",
        action="store_true",
        help="Override --n with the row count of "
             "outputs/genuine_recovery_dataset.json so the ablation has the "
             "exact same N as the main exp6 run.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        default=str(EXP6_DIR / "outputs" / "limo_ablation_dataset.json"),
    )
    args = ap.parse_args()

    src = Path(args.limo_dataset)
    if not src.exists():
        print(f"ERROR: LIMO dataset not found at {src}", file=sys.stderr)
        print("       Run parent train.sh once (or its make_limo_dataset.py step) to generate it.", file=sys.stderr)
        return 1

    rows = json.loads(src.read_text())
    if not isinstance(rows, list) or not rows:
        print(f"ERROR: {src} is empty or not a list", file=sys.stderr)
        return 1

    n = args.n
    if args.match_recovery_count:
        rec_path = EXP6_DIR / "outputs" / "genuine_recovery_dataset.json"
        if not rec_path.exists():
            print(f"ERROR: --match_recovery_count set but {rec_path} not found", file=sys.stderr)
            return 1
        n = len(json.loads(rec_path.read_text()))

    if n > len(rows):
        print(f"ERROR: requested n={n} but source only has {len(rows)} rows", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, n)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sampled, indent=2, ensure_ascii=False) + "\n")

    print(f"  source                  {src}  ({len(rows)} rows)")
    print(f"  sampled                 {n}  (seed={args.seed})")
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
