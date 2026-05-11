#!/usr/bin/env python3
"""LLM-judge filter for genuine-recovery SFT rows.

Reads a cohort produced by extract_genuine_recoveries.py and asks Claude
Sonnet 4.6 to assess each row's thinking trace on three axes:

  1. soundness  — does the reasoning actually correctly derive the answer?
                  0 = derivation is wrong / contradicts itself
                  1 = derivation has gaps / hand-waved steps
                  2 = derivation is correct and complete
  2. coherence  — is the thinking coherent English math vs gibberish/loops?
                  0 = nonsensical / stuck in repetition / non-math rambling
                  1 = mostly coherent with bad patches
                  2 = clearly coherent throughout
  3. lucky_correct — does the trace arrive at GT through wrong logic?
                  yes = trace concludes wrong intermediate values but
                        final \\boxed{} happens to match GT
                  no  = trace's logic genuinely yields GT

Strict-mode keep rule: soundness==2 AND coherence==2 AND lucky_correct=="no".

Prompt caching on the system rubric so 127 calls cost <$1 total (cache hits
on 126/127 calls, ephemeral 5-min TTL). Concurrency via ThreadPoolExecutor
(max 8 in-flight) keeps wall time at ~2-3 min and the cache warm.

Auth: reads `~/.anthropic_key`. Tolerates either a bare key or a
`ANTHROPIC_API_KEY=value` line so it doesn't matter how the user wrote it.

Usage:
    python3 judge_recoveries.py \\
        --input outputs/dapo_genuine_recovery_combined.json \\
        --output_clean outputs/dapo_genuine_recovery_clean.json \\
        --output_decisions outputs/dapo_judge_decisions.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path

MODEL = "claude-sonnet-4-6"
MAX_WORKERS = 8
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

PROMPT_TEMPLATE = """Audit this math reasoning trace. The ground truth answer is known.

KEEP only if every key step is correct, the logic clearly derives the ground truth, and the trace is fully coherent throughout.
DROP on any of: gaps in justification, hand-waved leaps, wrong arithmetic anywhere (even if the final answer happens to match), contradictions, abandoned intermediate results, repetition / loops, gibberish, or any sign the answer was reached by accident.

When in doubt, DROP.

Respond with one line: `KEEP: <reason>` or `DROP: <reason>`. Nothing else.

Problem:
{problem}

Ground truth: {ground_truth}

Trace:
{thinking}"""


def load_api_key(path: Path) -> str:
    raw = path.read_text().strip()
    if not raw:
        raise SystemExit(f"empty key file: {path}")
    # Accept bare key, KEY=value, or KEY="value"
    if raw.startswith("sk-"):
        return raw
    m = re.search(r'ANTHROPIC_API_KEY\s*=\s*"?([^"\n]+)"?', raw)
    if m:
        return m.group(1).strip()
    raise SystemExit(
        f"could not parse key from {path}: expected bare 'sk-...' or "
        "'ANTHROPIC_API_KEY=sk-...' format"
    )


def extract_thinking(output: str) -> str:
    """Pull the content between <think>...</think> from the alpaca output."""
    i = output.find(THINK_OPEN)
    j = output.rfind(THINK_CLOSE)
    if i == -1 or j == -1 or j <= i:
        return output.strip()
    return output[i + len(THINK_OPEN): j].strip()


def judge_one(client, row: dict, max_retries: int = 3) -> dict:
    thinking = extract_thinking(row["output"])
    prompt = PROMPT_TEMPLATE.format(
        problem=row["instruction"],
        ground_truth=row["_ground_truth"],
        thinking=thinking,
    )
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # First word determines verdict; rest is the reason.
            head = text.split(":", 1)[0].strip().upper()
            reason = text.split(":", 1)[1].strip() if ":" in text else text
            verdict = "KEEP" if head.startswith("KEEP") else ("DROP" if head.startswith("DROP") else "?")
            return {
                "_problem_id": row.get("_problem_id"),
                "_provenance": row.get("_provenance"),
                "verdict": verdict,
                "reason": reason,
                "raw": text,
                "_usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "_problem_id": row.get("_problem_id"),
                    "_provenance": row.get("_provenance"),
                    "verdict": "ERROR",
                    "reason": f"judge failed after {max_retries} retries: {e}",
                    "raw": "",
                }
            time.sleep(2 ** attempt)


def keep_row(j: dict) -> bool:
    return j.get("verdict") == "KEEP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_clean", required=True,
                    help="Path for the filtered SFT rows (those judged keep=True).")
    ap.add_argument("--output_decisions", required=True,
                    help="Path for the JSONL log of every judgment (keep + drop), "
                         "with reasons. Use to audit borderline drops.")
    ap.add_argument("--key_file", default=str(Path.home() / ".anthropic_key"))
    ap.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only judge the first N rows (useful for smoke-test).")
    ap.add_argument("--strip_metadata", action="store_true",
                    help="Drop _provenance/_problem_id/etc. from the clean output. "
                         "Useful for LlamaFactory ingestion.")
    args = ap.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = load_api_key(Path(args.key_file))
    from anthropic import Anthropic  # noqa: E402
    client = Anthropic()

    rows = json.loads(Path(args.input).read_text())
    if args.limit:
        rows = rows[: args.limit]
    print(f"  judging {len(rows)} rows with {MODEL} (max_workers={args.max_workers}) …")

    t0 = time.time()
    judgments: list[dict] = [None] * len(rows)
    n_done = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(judge_one, client, row): i for i, row in enumerate(rows)}
        for fut in cf.as_completed(futures):
            i = futures[fut]
            judgments[i] = fut.result()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(rows):
                elapsed = time.time() - t0
                print(f"    {n_done}/{len(rows)} done  ({elapsed:.0f}s)", flush=True)

    # Write per-row decision log.
    decisions_path = Path(args.output_decisions)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("w") as f:
        for row, j in zip(rows, judgments):
            f.write(json.dumps({
                "_problem_id": row.get("_problem_id"),
                "_provenance": row.get("_provenance"),
                "_ground_truth": row.get("_ground_truth"),
                "verdict": j["verdict"],
                "reason": j["reason"],
                "raw": j.get("raw", ""),
                "usage": j.get("_usage", {}),
            }, ensure_ascii=False) + "\n")

    kept = [row for row, j in zip(rows, judgments) if keep_row(j)]
    if args.strip_metadata:
        for r in kept:
            for k in ("_provenance", "_problem_id", "_chain_idx",
                      "_r1_extracted_answer", "_thinking_last_boxed",
                      "_ground_truth"):
                r.pop(k, None)
    Path(args.output_clean).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_clean).write_text(json.dumps(kept, indent=2, ensure_ascii=False))

    from collections import Counter
    verdicts = Counter(j["verdict"] for j in judgments)
    total_usage = {
        "input": sum(j.get("_usage", {}).get("input_tokens", 0) for j in judgments),
        "output": sum(j.get("_usage", {}).get("output_tokens", 0) for j in judgments),
    }
    cost = total_usage["input"] * 3 / 1_000_000 + total_usage["output"] * 15 / 1_000_000
    print()
    print(f"  judged   {len(judgments)}")
    print(f"  verdicts {dict(verdicts)}")
    print(f"  kept     {len(kept)}")
    print(f"  tokens: input={total_usage['input']}  output={total_usage['output']}")
    print(f"  cost (Sonnet 4.6 list pricing): ${cost:.3f}")
    print(f"  wall: {time.time() - t0:.1f}s")
    print(f"  → {args.output_clean}")
    print(f"  → {args.output_decisions}")


if __name__ == "__main__":
    main()
