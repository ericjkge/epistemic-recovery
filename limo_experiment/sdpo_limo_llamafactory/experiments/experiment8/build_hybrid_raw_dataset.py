#!/usr/bin/env python3
"""Rebuild the experiment 8 hybrid SFT dataset WITHOUT narrativization.

Motivation (2026-05-11): the narrativized dataset trained a LoRA that lost the
ability to commit to an answer. Even on easy AIME25 probes, the LoRA produced
13K-token traces with zero `\\boxed{}` while baseline solved in 2K. The diagnostic
ruled out long-context instability — the LoRA itself regressed. Hypothesis:
stripping all inline `\\boxed{}` and `**Final Answer**` markers from <think>
removed exactly the scaffolding the model needs to learn "this is when I land
on an answer."

This builder reconstructs the same 732-row composition (600 off-policy LIMO + 51
on-policy LIMO + 81 DAPO recovery) from the un-narrativized sources, preserving
intermediate `\\boxed{X}` anchors and recovery-row R1 commit blocks intact. The
only invariants enforced: each row's `output` ends with a single terminal
`\\boxed{}` after `</think>` (already true of all source rows).
"""
import argparse
import json
import re
import sys
from pathlib import Path

EXP8_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP8_DIR.parent.parent


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current_hybrid", default=str(EXP8_DIR / "outputs" / "hybrid_narrative_dataset.json"))
    ap.add_argument("--limo_src", default=str(PARENT_DIR / "limo_v2_sdpo.json"))
    ap.add_argument("--dapo_src", default=str(PARENT_DIR / "experiments/experiment5/outputs/dapo_genuine_recovery_clean.json"))
    ap.add_argument("--exp6_src", default=str(PARENT_DIR / "experiments/experiment6/outputs/genuine_recovery_dataset.json"))
    ap.add_argument("--output", default=str(EXP8_DIR / "outputs" / "hybrid_raw_dataset.json"))
    args = ap.parse_args()

    hyb = json.loads(Path(args.current_hybrid).read_text())
    limo_by = {norm(r["instruction"]): r for r in json.loads(Path(args.limo_src).read_text())}
    dapo_by = {norm(r["instruction"]): r for r in json.loads(Path(args.dapo_src).read_text())}
    exp6_by = {norm(r["instruction"]): r for r in json.loads(Path(args.exp6_src).read_text())}

    rebuilt = []
    miss = 0
    by_prov = {"offpolicy_limo_raw": 0, "onpolicy_limo_raw": 0, "onpolicy_dapo_raw": 0}
    for r in hyb:
        prov = r["_provenance"]
        inst = norm(r["instruction"])
        src_row = None
        new_prov = None
        if prov == "offpolicy_limo_narrative":
            src_row = limo_by.get(inst)
            new_prov = "offpolicy_limo_raw"
        elif prov == "onpolicy_limo_narrative":
            src_row = exp6_by.get(inst) or limo_by.get(inst)
            new_prov = "onpolicy_limo_raw"
        elif prov == "onpolicy_dapo_narrative":
            src_row = dapo_by.get(inst)
            new_prov = "onpolicy_dapo_raw"
        if src_row is None:
            miss += 1
            continue
        # Copy raw output; preserve original system + instruction.
        new_r = {
            "instruction": r["instruction"],
            "input": r.get("input", ""),
            "output": src_row["output"],
            "system": r.get("system", src_row.get("system", "")),
            "_provenance": new_prov,
        }
        rebuilt.append(new_r)
        by_prov[new_prov] += 1

    # Same shuffle seed as parent build for reproducibility.
    import random
    random.Random(42).shuffle(rebuilt)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n")

    # Audit: confirm exactly 1 \boxed{} after </think> in all rows.
    n_term_ok = 0
    for r in rebuilt:
        out = r["output"]
        if "</think>" in out:
            post = out.split("</think>", 1)[1]
            if len(re.findall(r"\\boxed\{", post)) == 1:
                n_term_ok += 1
    print(f"  total: {len(rebuilt)} rows  (missing={miss})")
    for k, v in by_prov.items():
        print(f"    {k}: {v}")
    print(f"  terminal \\boxed{{}} invariant: {n_term_ok}/{len(rebuilt)} rows")
    print(f"  → {args.output}")
    return 0 if n_term_ok == len(rebuilt) else 1


if __name__ == "__main__":
    sys.exit(main())
