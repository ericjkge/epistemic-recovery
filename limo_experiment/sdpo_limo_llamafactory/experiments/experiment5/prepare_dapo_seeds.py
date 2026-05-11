#!/usr/bin/env python3
"""Build a DAPO-Math-17k seed file for experiment 5's injection generator.

Why this exists (vs prepare_seed_problems.py): DAPO's row schema differs from
LIMO's flat {problem, answer}. Each row is

    prompt:        [{"role": "user", "content": "Solve the following ...\nRemember..."}]
    reward_model:  {"ground_truth": "42", "style": "rule-lighteval/MATH_v2"}

and the HF mirror is 1.79M rows because each of the ~17.4K unique problems is
replicated ~100x for RL rollouts. We dedupe by `content`, strip DAPO's
"Answer:"-style instruction wrapper (so generate_injections.py's USER_TEMPLATE
can re-wrap with the \\boxed{} instruction), filter to integer ground truths
(required by grade_int), shuffle deterministically, and write the first --max
to seed_problems.jsonl plus a sidecar `selected_indices.json` recording the
parquet row indices used. The sidecar lets a follow-up run pass
--exclude_indices to avoid re-sampling the same problems.

problem_id encodes the original parquet row index of the first appearance of
that unique prompt: `dapo_seed_{orig_idx:07d}`. That makes selected_indices.json
straightforward to consume and lets you cross-reference back to the parquet.

Usage:
    python3 prepare_dapo_seeds.py \\
        --max 4000 \\
        --seed 42 \\
        --output outputs/dapo_pass1/seed_problems.jsonl

    # Resume later, skipping problems already used:
    python3 prepare_dapo_seeds.py --max 4000 --seed 43 \\
        --exclude_indices outputs/dapo_pass1/selected_indices.json \\
        --output outputs/dapo_pass2/seed_problems.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

EXP5_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EXP5_DIR / "outputs" / "dapo_pass1" / "seed_problems.jsonl"

DAPO_LEADING_RE = re.compile(
    r"^Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: \$Answer "
    r"\(without quotes\) where \$Answer is the answer to the problem\.\s*\n*"
)
DAPO_TRAILING_RE = re.compile(
    r'\s*\n*Remember to put your answer on its own line after "Answer:"\.?\s*$'
)


def strip_dapo_wrapper(content: str) -> str:
    """Remove DAPO's instruction wrapper, leaving the bare problem text."""
    content = DAPO_LEADING_RE.sub("", content)
    content = DAPO_TRAILING_RE.sub("", content)
    return content.strip()


# CJK unified ideographs + Hangul + Hiragana/Katakana. Any single match
# disqualifies the row — DAPO has a chunk of Chinese-translated problems and
# the SDPO model is English-trained, so we want the bare ASCII/Latin subset.
_NON_ENGLISH_RE = re.compile(r"[぀-ヿ一-鿿가-힯]")


def is_english(text: str) -> bool:
    return _NON_ENGLISH_RE.search(text) is None


def coerce_int(val) -> int | None:
    if val is None:
        return None
    s = str(val).strip().replace(",", "")
    if not s:
        return None
    m = re.fullmatch(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="BytedTsinghua-SIA/DAPO-Math-17k",
                    help="HF dataset name. Default DAPO-Math-17k.")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--max", type=int, default=4000,
                    help="Number of seeds to write after dedupe/integer filter/shuffle.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed. Recording it (in the sidecar) makes the "
                         "selection reproducible across re-runs.")
    ap.add_argument("--exclude_indices", default=None,
                    help="Path to a selected_indices.json from a prior run. Any "
                         "parquet row indices listed there are skipped so a "
                         "follow-up pass collects new problems.")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    excluded: set[int] = set()
    if args.exclude_indices:
        with open(args.exclude_indices) as f:
            data = json.load(f)
        excluded = set(data.get("selected_indices") or data.get("indices") or [])
        print(f"  excluding {len(excluded)} indices from {args.exclude_indices}")

    from datasets import load_dataset
    print(f"  loading {args.source}:{args.hf_split} ...")
    ds = load_dataset(args.source, split=args.hf_split)
    print(f"  rows in dataset:           {len(ds)}")

    # ── Dedupe by prompt content, integer-filter ───────────────────────────
    # Walk once; remember the original row index of the first appearance of
    # each unique prompt. That index becomes part of the problem_id and the
    # sidecar — stable across re-runs even if HF shuffles internal storage.
    seen_content: set[str] = set()
    accepted: list[dict] = []      # {orig_idx, question, answer}
    n_dupe = 0
    n_no_int = 0
    n_non_english = 0
    n_excluded_sidecar = 0
    for i, row in enumerate(ds):
        prompt = row.get("prompt") or []
        if not isinstance(prompt, list) or not prompt:
            continue
        first = prompt[0]
        if not isinstance(first, dict):
            continue
        content = first.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if content in seen_content:
            n_dupe += 1
            continue
        seen_content.add(content)

        if i in excluded:
            n_excluded_sidecar += 1
            continue

        question = strip_dapo_wrapper(content)
        if not is_english(question):
            n_non_english += 1
            continue

        rm = row.get("reward_model") or {}
        gt = coerce_int(rm.get("ground_truth"))
        if gt is None:
            n_no_int += 1
            continue

        accepted.append({
            "orig_idx": i,
            "question": question,
            "answer": gt,
        })

    print(f"  unique prompts:              {len(seen_content)}")
    print(f"  duplicates skipped:          {n_dupe}")
    print(f"  uniques excluded by sidecar: {n_excluded_sidecar}")
    print(f"  uniques non-English:         {n_non_english}")
    print(f"  uniques without int GT:      {n_no_int}")
    print(f"  accepted (eligible pool):    {len(accepted)}")

    if not accepted:
        raise SystemExit("no seeds accepted after filters — nothing to write")

    # ── Shuffle and take --max ──────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    chosen = accepted[: args.max]
    print(f"  taking first {len(chosen)} after shuffle(seed={args.seed})")

    # ── Write seeds + sidecar ──────────────────────────────────────────────
    with open(out, "w") as f:
        for rec in chosen:
            f.write(json.dumps({
                "problem_id": f"dapo_seed_{rec['orig_idx']:07d}",
                "question": rec["question"],
                "answer": rec["answer"],
            }, ensure_ascii=False) + "\n")

    sidecar = out.parent / "selected_indices.json"
    with open(sidecar, "w") as f:
        json.dump({
            "source": args.source,
            "hf_split": args.hf_split,
            "shuffle_seed": args.seed,
            "filters": {"english_only": True, "integer_gt": True},
            "n_selected": len(chosen),
            "n_eligible_pool": len(accepted),
            "n_excluded_by_sidecar": n_excluded_sidecar,
            "selected_indices": sorted(rec["orig_idx"] for rec in chosen),
        }, f, indent=2)

    print(f"  → {out}")
    print(f"  → {sidecar}")


if __name__ == "__main__":
    main()
