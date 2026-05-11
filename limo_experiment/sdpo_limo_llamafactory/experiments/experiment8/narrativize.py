#!/usr/bin/env python3
"""Data-hygiene utility for experiment 8.

Enforces a single global rule on every alpaca-row's `output` field:

    ``\\boxed{...}`` and ``**Final Answer**`` appear EXACTLY ONCE per response,
    in the terminal commit AFTER ``</think>``. Inside ``<think>...</think>``,
    reasoning is pure prose narrative — no structural commit markers, no
    inline ``\\boxed{}``, no ``**Final Answer**``.

The two sources that violate this rule (and that this script fixes):
1. LIMO-v2 traces — human authors use ``\\boxed{intermediate}`` as inline LaTeX
   throughout solutions. The original wrapping in ``make_limo_dataset.py``
   only strips the LAST ``\\boxed{}``, leaving earlier ones inside ``<think>``.
2. Pass-4 on-policy recovery traces — the SDPO base model emits
   ``**Final Answer**\\n\\boxed{X}`` blocks inside thinking by habit. The
   injection protocol's ``reopen_thinking()`` strips the post-``</think>``
   content but leaves the pre-``</think>`` "Final Answer" block in place.

Transform pipeline (applied ONLY inside the <think>...</think> span):
1. Remove ``[ws]**Final Answer**[:?][ws]\\boxed{...}[ws]`` blocks
   (with brace-balanced matching for nested LaTeX like ``\\boxed{\\frac{a}{b}}``).
2. Replace any remaining inline ``\\boxed{X}`` → ``X`` (also brace-balanced).
   Empty ``\\boxed{}`` strings are PRESERVED — the model often quotes the system
   prompt's "put your final answer within \\boxed{}" verbatim, which is
   legitimate prose, not a commit.
3. Remove any leftover ``**Final Answer**`` markers.
4. Collapse runs of >2 blank lines to 2.

Post-``</think>`` content is left untouched (that's the SINGLE terminal commit,
which is the only place ``\\boxed{}`` is allowed).
"""
import argparse
import json
import re
import sys
from pathlib import Path

_BOXED_OPEN = "\\boxed{"

# Matches the structural prefix of "**Final Answer**\n\boxed{" with optional
# heading-style decoration (`### `, `#`, `**`). Used to find the start of a
# Final-Answer commit block; brace balancing then locates the closing `}`.
_FA_BLOCK_PREFIX_RE = re.compile(
    r"\n*\s*(?:#{1,6}\s+)?(?:\*\*)?Final\s+Answer(?:\*\*)?\s*:?\s*\n+\s*\\boxed\{",
    re.IGNORECASE,
)

# Same prefix variants, but followed by a `$$ ... $$` math-display block as
# the commit (a separate flavor common in LIMO traces). Handled as a single
# regex (no brace balancing needed — `$$...$$` is line-delimited).
_FA_MATHDISP_RE = re.compile(
    r"\n*\s*(?:#{1,6}\s+)?(?:\*\*)?Final\s+Answer(?:\*\*)?\s*:?\s*\n+\$\$\s*\n?[^$]+?\n?\s*\$\$\s*",
    re.IGNORECASE,
)

# Any leftover Final-Answer marker (no attached commit) — strip the marker
# but leave surrounding text. Covers `**Final Answer**`, `### Final Answer`,
# plain `Final Answer`, `Final Answer:`, etc.
#
# The `(?=\n|$)` lookahead requires the marker to be followed by a newline
# (or end-of-string). True structural markers always are; this guard prevents
# the regex from eating "Final Answer" inside narrative prose like
# `the **Final Answer** is X` (where a word follows after a space).
_FA_MARKER_RE = re.compile(
    r"(?:#{1,6}\s+)?(?:\*\*)?Final\s+Answer(?:\*\*)?\s*:?(?=\s*(?:\n|$))",
    re.IGNORECASE,
)

_BLANKS_RE = re.compile(r"\n{3,}")


def _walk_balanced_braces(text: str, open_idx: int) -> int:
    """Given text[open_idx] == '{', return index just past the matching '}'.
    Returns -1 if unbalanced (no matching close found)."""
    depth = 1
    j = open_idx + 1
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _remove_fa_commit_blocks(text: str) -> str:
    """Remove '**Final Answer**[:?]\\n+\\boxed{...}' blocks with brace balancing."""
    out = []
    i = 0
    while i < len(text):
        m = _FA_BLOCK_PREFIX_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        # m.end() is the position right after '\\boxed{'; the '{' is at m.end()-1.
        end_box = _walk_balanced_braces(text, m.end() - 1)
        if end_box < 0:
            # Unbalanced — fall back to leaving the marker text alone
            out.append(text[m.start():m.end()])
            i = m.end()
            continue
        # Skip trailing horizontal whitespace after the boxed close
        j = end_box
        while j < len(text) and text[j] in " \t":
            j += 1
        out.append("\n\n")
        i = j
    return "".join(out)


def _replace_inline_boxed(text: str) -> str:
    """Brace-balanced replace ``\\boxed{X}`` with ``X``. Empty ``\\boxed{}`` is
    preserved (it's quoted system-prompt text, not a commit)."""
    out = []
    i = 0
    L = len(text)
    while i < L:
        if text.startswith(_BOXED_OPEN, i):
            open_brace_idx = i + len(_BOXED_OPEN) - 1
            end_box = _walk_balanced_braces(text, open_brace_idx)
            if end_box < 0:
                out.append(text[i])
                i += 1
                continue
            content = text[i + len(_BOXED_OPEN) : end_box - 1]
            if content == "":
                out.append(text[i:end_box])
            else:
                out.append(content)
            i = end_box
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def narrativize_assistant_output(text: str) -> str:
    """Rewrite a full alpaca `output` value so the <think> span is narrative-only.
    Post-</think> tail is preserved verbatim."""
    close_idx = text.rfind("</think>")
    if close_idx < 0:
        thinking, post = text, ""
    else:
        thinking, post = text[:close_idx], text[close_idx:]

    thinking = _remove_fa_commit_blocks(thinking)
    thinking = _FA_MATHDISP_RE.sub("\n\n", thinking)
    thinking = _replace_inline_boxed(thinking)
    thinking = _FA_MARKER_RE.sub("", thinking)
    thinking = _BLANKS_RE.sub("\n\n", thinking).rstrip() + "\n"

    return thinking + post


def count_boxed_in_think(text: str) -> int:
    """Count non-empty ``\\boxed{...}`` instances inside <think>.
    Skips empty ``\\boxed{}`` (legitimate quoted text)."""
    close_idx = text.rfind("</think>")
    span = text[:close_idx] if close_idx >= 0 else text
    n = 0
    i = 0
    while i < len(span):
        j = span.find(_BOXED_OPEN, i)
        if j < 0:
            break
        end_box = _walk_balanced_braces(span, j + len(_BOXED_OPEN) - 1)
        if end_box < 0:
            i = j + 1
            continue
        content = span[j + len(_BOXED_OPEN) : end_box - 1]
        if content != "":
            n += 1
        i = end_box
    return n


def count_in_think(text: str, needle: str) -> int:
    close_idx = text.rfind("</think>")
    return text[:close_idx].count(needle) if close_idx >= 0 else text.count(needle)


def transform_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = {
        "n_rows": len(rows),
        "boxed_in_think_before": 0,
        "boxed_in_think_after": 0,
        "fa_in_think_before": 0,
        "fa_in_think_after": 0,
        "chars_before": 0,
        "chars_after": 0,
        "post_boxed_unchanged": 0,
    }
    out = []
    for r in rows:
        text = r["output"]
        stats["boxed_in_think_before"] += count_boxed_in_think(text)
        stats["fa_in_think_before"] += count_in_think(text, "Final Answer")
        stats["chars_before"] += len(text)

        new_text = narrativize_assistant_output(text)
        stats["boxed_in_think_after"] += count_boxed_in_think(new_text)
        stats["fa_in_think_after"] += count_in_think(new_text, "Final Answer")
        stats["chars_after"] += len(new_text)

        if text.rfind("</think>") >= 0 and new_text.rfind("</think>") >= 0:
            if text[text.rfind("</think>") :] == new_text[new_text.rfind("</think>") :]:
                stats["post_boxed_unchanged"] += 1

        new_r = dict(r)
        new_r["output"] = new_text
        out.append(new_r)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row still has non-empty \\boxed{} or 'Final Answer' inside <think>.",
    )
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(src.read_text())
    new_rows, stats = transform_rows(rows)
    dst.write_text(json.dumps(new_rows, indent=2, ensure_ascii=False) + "\n")

    print(f"  source                 {src}  ({stats['n_rows']} rows)")
    print(f"  output                 {dst}")
    print(f"  chars                  {stats['chars_before']:>10d} → {stats['chars_after']:>10d}  "
          f"(Δ {stats['chars_after']-stats['chars_before']:+d})")
    print(f"  \\boxed{{X}} in <think>    "
          f"{stats['boxed_in_think_before']:>10d} → {stats['boxed_in_think_after']:>10d}  (empty \\boxed{{}} excluded)")
    print(f"  'Final Answer' in <think> "
          f"{stats['fa_in_think_before']:>8d} → {stats['fa_in_think_after']:>8d}")
    print(f"  post-</think> unchanged {stats['post_boxed_unchanged']}/{stats['n_rows']}")

    if args.strict and (stats["boxed_in_think_after"] > 0 or stats["fa_in_think_after"] > 0):
        print("STRICT MODE FAIL: residual markers in <think>", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
