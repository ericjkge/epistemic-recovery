#!/usr/bin/env python3
"""One-shot: rebuild the on-disk hybrid dataset with the new terminal-commit policy.

Reads `outputs/hybrid_narrative_dataset.json` (the 732-row v1, narrativized to
strict prose-only thinking) and re-narrativizes each row with the now-default
`preserve_terminal_commit=True` option. The pre-existing narrativization steps
are idempotent on already-narrativized rows (no FA blocks, no non-empty inline
\\boxed{}); the only new content is a single `**Final Answer**\\n\\boxed{X}`
block injected immediately before `</think>`, with X extracted from each row's
post-`</think>` final-answer line.

Outputs:
  - outputs/hybrid_narrative_dataset_v2.json        (with `_provenance` metadata)
  - outputs/hybrid_narrative_dataset_v2_train.json  (metadata-stripped for LF)

Register the train file as `hybrid_narrative_exp8_v2` in `dataset_info.json`.
"""
import json
import sys
from collections import Counter
from pathlib import Path

EXP8 = Path(__file__).resolve().parent

sys.path.insert(0, str(EXP8))
from narrativize import (  # noqa: E402
    narrativize_assistant_output,
    count_boxed_in_think,
    count_in_think,
)


def main():
    src = EXP8 / "outputs" / "hybrid_narrative_dataset.json"
    dst_full = EXP8 / "outputs" / "hybrid_narrative_dataset_v2.json"
    dst_train = EXP8 / "outputs" / "hybrid_narrative_dataset_v2_train.json"

    rows = json.loads(src.read_text())
    print(f"loaded {len(rows)} rows from {src}")

    new_rows = []
    prov_counts = Counter()
    box_before = 0
    fa_before = 0
    box_after = 0
    fa_after = 0
    no_post_commit = 0
    for r in rows:
        text = r["output"]
        box_before += count_boxed_in_think(text)
        fa_before += count_in_think(text, "Final Answer")

        new_text = narrativize_assistant_output(text, preserve_terminal_commit=True)
        nb = count_boxed_in_think(new_text)
        nf = count_in_think(new_text, "Final Answer")
        box_after += nb
        fa_after += nf
        if nb == 0:
            no_post_commit += 1

        new_r = dict(r)
        new_r["output"] = new_text
        new_rows.append(new_r)
        prov_counts[r.get("_provenance", "?")] += 1

    print(f"  by provenance: {dict(prov_counts)}")
    print(f"  \\boxed in <think>:    {box_before:>5d} → {box_after:>5d}   (expected after = {len(rows)})")
    print(f"  Final Answer in <think>: {fa_before:>5d} → {fa_after:>5d}   (expected after = {len(rows)})")
    print(f"  rows with no terminal commit added (no boxed in post): {no_post_commit}")

    if box_after != len(rows) or fa_after != len(rows):
        print(
            "ERROR: expected exactly 1 terminal commit per row after v2 narrativization",
            file=sys.stderr,
        )
        return 2

    dst_full.write_text(json.dumps(new_rows, indent=2, ensure_ascii=False) + "\n")
    train_rows = [{k: v for k, v in r.items() if k != "_provenance"} for r in new_rows]
    dst_train.write_text(json.dumps(train_rows, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {len(new_rows)} rows to {dst_full}")
    print(f"wrote {len(train_rows)} rows to {dst_train}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
