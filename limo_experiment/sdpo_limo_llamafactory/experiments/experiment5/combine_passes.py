#!/usr/bin/env python3
"""Combine multiple injection_traces.jsonl files into one by concatenating
chains per problem.

Use case: pass@4 dataset assembled from a pass@1 run + a pass@3 run (or
multiple seeded re-runs). Each input run produced one record per problem
with a `chains` list; this script merges them so each problem has all
chains under a single record. Downstream make_injection_dataset.py and
analyze_injection_epistemic.py work on the merged file unchanged.

Behaviour:
  - Records are matched by `problem_id`. A problem present in only some
    inputs keeps the chains it has.
  - Per-record fields (`question`, `ground_truth`, etc.) are copied from
    the first input that has the problem; later inputs only contribute
    `chains` (any disagreement on `ground_truth` is logged and the first
    value wins).
  - `chain_idx` is renumbered 0..(K-1) globally across the merged chains
    to keep make_injection_dataset.py's per-row indexing consistent.
  - `num_chains`, `any_correct`, `min_correct_round` are recomputed.
  - Stats sidecar (`stats.json`) is also written, summarising the merged
    pass@K rates per round. The merged max_rounds is the max across
    inputs (rounds that didn't run in some inputs are simply absent for
    those chains).

Usage:
    python3 combine_passes.py \\
        --inputs outputs/pass1/injection_traces.jsonl \\
                 outputs/pass3/injection_traces.jsonl \\
        --output_dir outputs/pass4
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

EXP5_DIR = Path(__file__).resolve().parent


def load_traces(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_stats(traces_path: Path) -> dict | None:
    stats_path = traces_path.parent / "stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Two or more injection_traces.jsonl files to merge.")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write the merged injection_traces.jsonl + stats.json.")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("--inputs needs at least 2 files to combine")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_traces = out_dir / "injection_traces.jsonl"
    out_stats = out_dir / "stats.json"

    # Merge in input order so the first file's per-record metadata wins.
    merged: "OrderedDict[str, dict]" = OrderedDict()
    input_metadata: list[dict] = []  # one entry per input for stats sidecar

    for src_idx, src in enumerate(args.inputs):
        src_path = Path(src)
        rows = load_traces(src_path)
        src_stats = load_stats(src_path) or {}
        input_metadata.append({
            "path": str(src_path),
            "n_problems_seen": len(rows),
            "injection_phrase": src_stats.get("injection_phrase"),
            "num_chains": src_stats.get("num_chains"),
            "max_rounds": src_stats.get("max_rounds"),
            "model_path": src_stats.get("model_path"),
            "seed": (src_stats.get("args") or {}).get("seed"),
        })

        for rec in rows:
            pid = rec["problem_id"]
            if pid not in merged:
                merged[pid] = {
                    "problem_id": pid,
                    "question": rec["question"],
                    "ground_truth": rec["ground_truth"],
                    "chains": [],
                }
            else:
                # Sanity check: all inputs should agree on ground truth.
                if merged[pid]["ground_truth"] != rec["ground_truth"]:
                    print(f"  WARNING: {pid} ground_truth mismatch "
                          f"{merged[pid]['ground_truth']!r} vs {rec['ground_truth']!r} "
                          f"(input {src_idx}); keeping first.")
            merged[pid]["chains"].extend(rec.get("chains", []))

    # Renumber chain_idx globally per problem; recompute summary fields.
    n_problems = len(merged)
    n_problems_any_correct = 0
    pass_at_k_total = 0
    pass_at_k_at_round_1 = 0
    pass_at_k_at_round_2 = 0
    chain_pass_at_1_total = 0
    chains_total = 0

    with open(out_traces, "w") as f:
        for rec in merged.values():
            chains = rec["chains"]
            for i, ch in enumerate(chains):
                ch["chain_idx"] = i
            any_correct = any(ch.get("final_correct") for ch in chains)
            correct_rounds = [
                ch.get("final_round", 0) for ch in chains if ch.get("final_correct")
            ]
            min_correct_round = min(correct_rounds) if correct_rounds else 0
            rec["num_chains"] = len(chains)
            rec["any_correct"] = any_correct
            rec["min_correct_round"] = min_correct_round

            # Round-bucketed pass@K accounting (a problem is "passed by round r"
            # if any chain reached final_correct=True with final_round<=r).
            if any_correct:
                n_problems_any_correct += 1
                if any(
                    ch.get("final_correct") and ch.get("final_round", 99) <= 1
                    for ch in chains
                ):
                    pass_at_k_at_round_1 += 1
                if any(
                    ch.get("final_correct") and ch.get("final_round", 99) <= 2
                    for ch in chains
                ):
                    pass_at_k_at_round_2 += 1
            pass_at_k_total = n_problems_any_correct  # cumulative final = whole-trace
            chains_total += len(chains)
            chain_pass_at_1_total += sum(1 for ch in chains if ch.get("final_correct"))

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    K_combined = max((r["num_chains"] for r in merged.values()), default=0)
    stats = {
        "n_problems": n_problems,
        "num_chains_max": K_combined,
        "num_chains_uniform": all(
            r["num_chains"] == K_combined for r in merged.values()
        ),
        "pass_at_K_round_correct": [pass_at_k_at_round_1, pass_at_k_at_round_2],
        "pass_at_K_round_rate": [
            pass_at_k_at_round_1 / n_problems if n_problems else 0.0,
            pass_at_k_at_round_2 / n_problems if n_problems else 0.0,
        ],
        "n_problems_any_correct": n_problems_any_correct,
        "chain_pass_at_1_total": chain_pass_at_1_total,
        "chains_total": chains_total,
        "chain_pass_at_1_rate": (
            chain_pass_at_1_total / chains_total if chains_total else 0.0
        ),
        "inputs": input_metadata,
    }
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  inputs:                   {len(args.inputs)} files")
    for m in input_metadata:
        print(f"    - {m['path']}  (chains/problem={m['num_chains']}, seed={m['seed']})")
    print(f"  merged problems:          {n_problems}")
    print(f"  merged chains/problem:    {K_combined}{' (uniform)' if stats['num_chains_uniform'] else ' (uneven across problems)'}")
    print(f"  pass@K (any chain) by R:  R1={pass_at_k_at_round_1}/{n_problems}  R2={pass_at_k_at_round_2}/{n_problems}")
    print(f"  per-chain pass@1:         {chain_pass_at_1_total}/{chains_total} "
          f"({stats['chain_pass_at_1_rate']:.3f})")
    print(f"  → {out_traces}")
    print(f"  → {out_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
