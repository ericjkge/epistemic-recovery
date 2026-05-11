#!/usr/bin/env python3
"""Build the experiment 8 hybrid SFT dataset under strict data-hygiene.

Pipeline:
  1. On-policy slice = a strict-clean recovery cohort (default 56 rows from
     `experiments/experiment8/outputs/clean_recovery_dataset.json`; override
     with --recovery_clean to point at any alpaca-format file, e.g. the
     experiment 6 genuine cohort at
     `experiments/experiment6/outputs/genuine_recovery_dataset.json`),
     narrativized via `narrativize_assistant_output`.
  2. Off-policy slice = top-N (default 600) LIMO-v2 rows by epistemic density
     under a length-percentile cap, also narrativized.
  3. Concatenated, shuffled (seed=42).

Hygiene rule enforced on every row:
    `\\boxed{}` and `**Final Answer**` appear EXACTLY ONCE per response, in
    the terminal commit after `</think>`. Inside `<think>...</think>`, the
    trace is pure prose — no boxed values, no Final-Answer markers.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

EXP8_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP8_DIR.parent.parent  # sdpo_limo_llamafactory/

sys.path.insert(0, str(PARENT_DIR / "eval"))
from analyze_epistemic import EPISTEMIC_TOKENS, TOKEN_REGEXES, extract_thinking_span  # noqa: E402

# Local import — same dir
sys.path.insert(0, str(EXP8_DIR))
from narrativize import narrativize_assistant_output, count_in_think, count_boxed_in_think  # noqa: E402


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().lower()


def score_row(row: dict) -> tuple[int, int, float]:
    thinking, _ = extract_thinking_span(row["output"])
    words = len(thinking.split())
    total = sum(len(rgx.findall(thinking)) for rgx in TOKEN_REGEXES.values())
    dens = (1000.0 * total / words) if words else 0.0
    return words, total, dens


def select_limo(limo_rows, n_target, length_pct_cap, excluded_questions, seed):
    """Reuse experiment 8's selection logic: length-cap then density rank."""
    # Filter out rows that overlap excluded questions (optional safety net).
    candidates = [
        r for r in limo_rows
        if normalize_question(r["instruction"]) not in excluded_questions
    ]
    scored = []
    for r in candidates:
        words, total, dens = score_row(r)
        if words == 0:
            continue
        scored.append((words, total, dens, r))
    lengths = sorted(s[0] for s in scored)
    p = max(0.0, min(100.0, length_pct_cap))
    cutoff_idx = max(1, int(round((p / 100.0) * len(lengths)))) - 1
    cutoff_idx = min(cutoff_idx, len(lengths) - 1)
    length_cutoff = lengths[cutoff_idx]
    under_cap = [s for s in scored if s[0] <= length_cutoff]
    under_cap.sort(key=lambda s: (-s[2], s[0]))
    if n_target > len(under_cap):
        return [r for _, _, _, r in under_cap], {
            "warning": f"requested {n_target} but only {len(under_cap)} under cap",
            "length_cutoff": length_cutoff,
            "candidates_total": len(scored),
        }
    return [r for _, _, _, r in under_cap[:n_target]], {
        "length_cutoff": length_cutoff,
        "candidates_total": len(scored),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recovery_clean",
        default=str(EXP8_DIR / "outputs" / "clean_recovery_dataset.json"),
        help="Strict-clean recovery cohort in alpaca format (default: 56-row "
             "experiment 8 cohort). Override to point at the experiment 6 "
             "genuine cohort or a freshly-combined pass cohort.",
    )
    ap.add_argument(
        "--limo_dataset",
        default=str(PARENT_DIR / "limo_v2_sdpo.json"),
    )
    ap.add_argument("--n_limo", type=int, default=600)
    ap.add_argument("--length_pct_cap", type=float, default=75.0)
    ap.add_argument(
        "--exclude_overlap",
        action="store_true",
        help="Drop LIMO rows whose question equals a recovery row's question.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        default=str(EXP8_DIR / "outputs" / "hybrid_narrative_dataset.json"),
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row still has \\boxed{} or 'Final Answer' inside <think>.",
    )
    args = ap.parse_args()

    rec_path = Path(args.recovery_clean)
    limo_path = Path(args.limo_dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rec_path.exists():
        print(f"ERROR: recovery cohort not found: {rec_path}", file=sys.stderr)
        print("       Either build the strict-clean cohort first (see NOTES.md → 'Growing the", file=sys.stderr)
        print("       clean recovery cohort'), or fall back to the experiment 6 genuine cohort:", file=sys.stderr)
        print("         --recovery_clean experiments/experiment6/outputs/genuine_recovery_dataset.json", file=sys.stderr)
        return 1
    if not limo_path.exists():
        print(f"ERROR: LIMO dataset not found: {limo_path}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)

    # ── Recovery cohort: load, narrativize, tag ────────────────────────────────
    recovery_rows = json.loads(rec_path.read_text())
    rec_questions = {normalize_question(r["instruction"]) for r in recovery_rows}
    rec_box_before = sum(count_boxed_in_think(r["output"]) for r in recovery_rows)
    rec_fa_before = sum(count_in_think(r["output"], "Final Answer") for r in recovery_rows)

    rec_narr = []
    for r in recovery_rows:
        new_r = {k: v for k, v in r.items() if k not in ("_provenance", "_phrase_rewritten")}
        new_r["output"] = narrativize_assistant_output(r["output"])
        new_r["_provenance"] = "onpolicy_recovery_narrative"
        rec_narr.append(new_r)
    rec_box_after = sum(count_boxed_in_think(r["output"]) for r in rec_narr)
    rec_fa_after = sum(count_in_think(r["output"], "Final Answer") for r in rec_narr)

    # ── LIMO selection (same as exp 8) then narrativize ───────────────────────
    limo_rows = json.loads(limo_path.read_text())
    excluded = rec_questions if args.exclude_overlap else set()
    limo_selected, sel_info = select_limo(limo_rows, args.n_limo, args.length_pct_cap, excluded, args.seed)

    limo_box_before = sum(count_boxed_in_think(r["output"]) for r in limo_selected)
    limo_fa_before = sum(count_in_think(r["output"], "Final Answer") for r in limo_selected)

    limo_narr = []
    for r in limo_selected:
        new_r = dict(r)
        new_r["output"] = narrativize_assistant_output(r["output"])
        new_r["_provenance"] = "offpolicy_limo_narrative"
        limo_narr.append(new_r)
    limo_box_after = sum(count_boxed_in_think(r["output"]) for r in limo_narr)
    limo_fa_after = sum(count_in_think(r["output"], "Final Answer") for r in limo_narr)

    # ── Merge + shuffle ───────────────────────────────────────────────────────
    merged = rec_narr + limo_narr
    rng.shuffle(merged)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")

    print(f"  recovery cohort        {rec_path}  ({len(recovery_rows)} rows)")
    print(f"    \\boxed{{}} in <think>    {rec_box_before:>5d} → {rec_box_after:>5d}")
    print(f"    'Final Answer' in <think> {rec_fa_before:>5d} → {rec_fa_after:>5d}")
    print()
    print(f"  LIMO source            {limo_path}  ({len(limo_rows)} rows)")
    print(f"    selected             {len(limo_narr)}  (length_pct_cap={args.length_pct_cap}, "
          f"length_cutoff={sel_info.get('length_cutoff','?')} words)")
    print(f"    \\boxed{{}} in <think>    {limo_box_before:>5d} → {limo_box_after:>5d}")
    print(f"    'Final Answer' in <think> {limo_fa_before:>5d} → {limo_fa_after:>5d}")
    if "warning" in sel_info:
        print(f"    WARNING: {sel_info['warning']}")
    print()
    print(f"  TOTAL ROWS             {len(merged)}")
    print(f"  → {out_path}")

    if args.strict:
        if rec_box_after > 0 or rec_fa_after > 0 or limo_box_after > 0 or limo_fa_after > 0:
            print("STRICT MODE FAIL: residual markers in <think>", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
