#!/usr/bin/env python3
"""Post-hoc epistemic-token analysis over results/ JSON files from evaluate_aime.py.

Counts the Kim et al. 10-token set inside each generation's <think>...</think> span,
case-insensitive, whole-word only.

Headline metrics:
  - avg epistemic tokens per response (sum of all 10 tokens' counts, normalized by
    n_samples) — lower noise than per-token counts
  - any-correct rate (pass@N): fraction of problems with ≥1 correct sample, read
    from the JSON's `any_correct_rate` field

Outputs:
  - results/epistemic_summary.csv                per-(model, benchmark) stats
  - results/epistemic_per_response.png           bar chart of the headline count
  - results/accuracy_comparison.png              per-benchmark any-correct bars
  - results/epistemic_comparison.png             per-token breakdown (diagnostic)
  - results/length_vs_accuracy.png               scatter (diagnostic)

Run from the sdpo_limo_llamafactory/ parent directory:
    source .venv-train/bin/activate
    python eval/analyze_epistemic.py --results_dir results
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPISTEMIC_TOKENS = [
    "wait", "hmm", "perhaps", "maybe", "actually",
    "alternatively", "seems", "might", "likely", "check",
]
TOKEN_REGEXES = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in EPISTEMIC_TOKENS}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_thinking_span(text: str):
    """Return (thinking_text, had_explicit_block: bool).

    Three thinking-mode patterns we see in the wild:

      1. Closed `<think>...</think>` with content → return inner. Standard
         Qwen3 thinking-mode output.
      2. Empty `<think></think>` followed by reasoning outside the tags →
         return the post-`</think>` portion up to the final `\\boxed{}`.
         LoRA adapters trained on raw LIMO traces (no `<think>` wrapping)
         learn to emit an empty think block and put all reasoning after it.
         Without this branch, epistemic counts come back as 0 even though
         the response is full of "wait/hmm/perhaps/...".
      3. Open `<think>` without closing (truncated at max_tokens) → return
         the whole post-tag remainder. Otherwise the analyzer would silently
         fall back to counting over the full response, inflating counts
         relative to closed-block baseline generations.
    """
    m = THINK_RE.search(text)
    if m:
        inner = m.group(1)
        # Case 2: empty/near-empty think with real content after — use post-tag.
        if len(inner.strip()) < 10 and len(text) - m.end() > 100:
            tail = text[m.end():]
            # Trim the final `\boxed{...}` answer line so thinking_length is
            # comparable to closed-block baselines (which exclude the answer).
            box_idx = tail.rfind("\\boxed{")
            if box_idx > 0:
                # Walk back to the start of the line containing \boxed{.
                line_start = tail.rfind("\n", 0, box_idx)
                tail = tail[: line_start if line_start != -1 else box_idx]
            return tail, True
        return inner, True
    if "<think>" in text:
        return text.split("<think>", 1)[1], True
    return text, False


def count_tokens(text: str) -> dict:
    return {t: len(rgx.findall(text)) for t, rgx in TOKEN_REGEXES.items()}


def words_len(text: str) -> int:
    # Cheap proxy for length when token IDs aren't available; the per-sample token-id
    # counts from vLLM are used where present.
    return len(text.split())


def parse_label(filename: str):
    """results/{label}_{benchmark}.json → (label, benchmark)."""
    stem = Path(filename).stem
    for bench in ("aime24", "aime25"):
        suffix = f"_{bench}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], bench
    return stem, "unknown"


def aggregate(results_dir: Path):
    rows = []
    no_think_warning = []

    for path in sorted(results_dir.glob("*_aime*.json")):
        label, bench = parse_label(path.name)
        with open(path) as f:
            payload = json.load(f)

        results = payload["results"]
        n_sampling = payload.get("n_sampling", len(results[0]["correctness"]) if results else 1)

        # acc@n: mean over all (problem, sample) pairs — estimates pass@1
        flat_correct = [c for r in results for c in r["correctness"]]
        acc = sum(flat_correct) / len(flat_correct) if flat_correct else 0.0

        # any-correct rate = pass@n: fraction of problems with ≥1 correct sample
        any_correct = payload.get("any_correct_rate")
        if any_correct is None:
            any_correct = (
                sum(1 for r in results if any(r["correctness"])) / len(results)
                if results else 0.0
            )

        total_resp_tokens = 0
        total_think_chars = 0
        total_resp_chars = 0
        n_samples = 0
        n_no_think = 0
        token_totals = defaultdict(int)

        for r in results:
            for gen, length in zip(r["generations"], r["response_lengths"]):
                think_text, had_block = extract_thinking_span(gen)
                if not had_block:
                    n_no_think += 1
                counts = count_tokens(think_text)
                for t, c in counts.items():
                    token_totals[t] += c
                total_resp_tokens += length
                total_resp_chars += len(gen)
                total_think_chars += len(think_text)
                n_samples += 1

        if n_no_think:
            no_think_warning.append((label, bench, n_no_think, n_samples))

        avg_resp_len = total_resp_tokens / n_samples if n_samples else 0
        # Approximate avg thinking length in tokens via char ratio (rough but cheap).
        avg_think_len = (
            avg_resp_len * (total_think_chars / total_resp_chars) if total_resp_chars else 0
        )
        total_epistemic_per_resp = (
            sum(token_totals.values()) / n_samples if n_samples else 0
        )

        # Standard errors.
        # SE of any-correct rate: each of n_problems is one Bernoulli trial.
        n_problems = len(results)
        se_any_correct = (
            math.sqrt(any_correct * (1 - any_correct) / n_problems) if n_problems > 0 else 0.0
        )
        # SE of acc@N: each of n_problems*n_sampling is one Bernoulli trial.
        n_total = n_problems * n_sampling
        se_acc = math.sqrt(acc * (1 - acc) / n_total) if n_total > 0 else 0.0

        row = {
            "model": label,
            "benchmark": bench,
            f"acc@{n_sampling}": round(acc, 4),
            f"se_acc@{n_sampling}": round(se_acc, 4),
            "any_correct_rate": round(any_correct, 4),
            "se_any_correct_rate": round(se_any_correct, 4),
            "avg_response_length": round(avg_resp_len, 1),
            "avg_thinking_length": round(avg_think_len, 1),
            "avg_epistemic_per_response": round(total_epistemic_per_resp, 3),
            "n_problems": n_problems,
            "n_samples": n_samples,
            "n_sampling": n_sampling,
        }
        for t in EPISTEMIC_TOKENS:
            row[t] = round(token_totals[t] / n_samples, 3) if n_samples else 0.0
        rows.append(row)

    return rows, no_think_warning


def write_csv(rows, out_csv: Path):
    if not rows:
        print("No rows to write.")
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")


def print_table(rows):
    if not rows:
        return
    n_sampling = rows[0].get("n_sampling", 16)
    cols = ["model", "benchmark", f"acc@{n_sampling}", f"se_acc@{n_sampling}",
            "any_correct_rate", "se_any_correct_rate",
            "avg_response_length", "avg_epistemic_per_response"]
    # Only include columns that exist in the rows.
    cols = [c for c in cols if c in rows[0]]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def plot_per_token_bars(rows, out_png: Path):
    """Horizontal bars, one panel per benchmark, grouped by model."""
    benches = sorted({r["benchmark"] for r in rows})
    models = sorted({r["model"] for r in rows})
    if not benches or not models:
        return

    fig, axes = plt.subplots(1, len(benches), figsize=(6 * len(benches), max(4, 0.4 * len(EPISTEMIC_TOKENS) * len(models))), sharey=True)
    if len(benches) == 1:
        axes = [axes]

    bar_h = 0.8 / max(len(models), 1)
    y_positions = list(range(len(EPISTEMIC_TOKENS)))

    for ax, bench in zip(axes, benches):
        bench_rows = {r["model"]: r for r in rows if r["benchmark"] == bench}
        for mi, model in enumerate(models):
            row = bench_rows.get(model)
            if row is None:
                continue
            offsets = [y + (mi - len(models) / 2) * bar_h + bar_h / 2 for y in y_positions]
            values = [row[t] for t in EPISTEMIC_TOKENS]
            ax.barh(offsets, values, height=bar_h, label=model)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(EPISTEMIC_TOKENS)
        ax.set_xlabel("avg occurrences per response (in <think>)")
        ax.set_title(bench)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def _panel_bar(ax, labels, values, title, ylabel, ylim=None, value_fmt="{:.3f}", errors=None):
    """One panel: one bar per model label, with values annotated above each bar.

    errors: optional list of ±1 SE values for vertical error bars.
    """
    colors = plt.cm.tab10.colors
    bars = ax.bar(labels, values, color=[colors[i % len(colors)] for i in range(len(labels))])
    if errors is not None:
        xs = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        ax.errorbar(xs, values, yerr=errors, fmt="none", color="black",
                    capsize=5, capthick=1.5, linewidth=1.5, zorder=5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, v in zip(bars, values):
        ax.annotate(value_fmt.format(v),
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")


def plot_any_correct_bars(rows, out_png: Path):
    """One panel per benchmark: bars of any-correct rate (pass@N) across models."""
    benches = sorted({r["benchmark"] for r in rows})
    models_ordered = _ordered_models(rows)
    if not benches or not models_ordered:
        return

    fig, axes = plt.subplots(1, len(benches), figsize=(5 * len(benches), 5), sharey=True)
    if len(benches) == 1:
        axes = [axes]

    n_sampling = rows[0].get("n_sampling", "N")
    for ax, bench in zip(axes, benches):
        bench_rows = {r["model"]: r for r in rows if r["benchmark"] == bench}
        labels = [m for m in models_ordered if m in bench_rows]
        values = [bench_rows[m]["any_correct_rate"] for m in labels]
        errors = [bench_rows[m].get("se_any_correct_rate", 0.0) for m in labels]
        _panel_bar(ax, labels, values,
                   title=bench,
                   ylabel=f"any-correct rate (pass@{n_sampling})",
                   ylim=(0, 1.05),
                   errors=errors)

    fig.suptitle(f"Per-problem pass rate across {n_sampling} samples", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def plot_epistemic_per_response_bars(rows, out_png: Path):
    """One panel per benchmark: bars of avg epistemic tokens per response."""
    benches = sorted({r["benchmark"] for r in rows})
    models_ordered = _ordered_models(rows)
    if not benches or not models_ordered:
        return

    fig, axes = plt.subplots(1, len(benches), figsize=(5 * len(benches), 5), sharey=True)
    if len(benches) == 1:
        axes = [axes]

    ymax = max(r["avg_epistemic_per_response"] for r in rows) * 1.2

    for ax, bench in zip(axes, benches):
        bench_rows = {r["model"]: r for r in rows if r["benchmark"] == bench}
        labels = [m for m in models_ordered if m in bench_rows]
        values = [bench_rows[m]["avg_epistemic_per_response"] for m in labels]
        _panel_bar(ax, labels, values,
                   title=bench,
                   ylabel="avg epistemic tokens per response",
                   ylim=(0, ymax),
                   value_fmt="{:.1f}")

    fig.suptitle("Epistemic verbalization: avg per-response count of the 10-token set", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _ordered_models(rows):
    """Preferred model order (baseline first, lora next, pretrained last if present)."""
    seen = []
    for prefer in ("baseline", "lora", "pretrained"):
        for r in rows:
            if prefer in r["model"] and r["model"] not in seen:
                seen.append(r["model"])
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    return seen


def plot_length_vs_accuracy(rows, out_png: Path):
    benches = sorted({r["benchmark"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = {"aime24": "o", "aime25": "s"}
    n_sampling = rows[0].get("n_sampling", "N") if rows else "N"
    for r in rows:
        ax.scatter(r["avg_response_length"], r["any_correct_rate"],
                   marker=markers.get(r["benchmark"], "^"), s=80,
                   label=f"{r['model']} / {r['benchmark']}")
        ax.annotate(r["model"], (r["avg_response_length"], r["any_correct_rate"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("avg response length (tokens)")
    ax.set_ylabel(f"any-correct rate (pass@{n_sampling})")
    ax.set_title("Length vs any-correct rate across models")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def qualitative_summary(rows):
    """Print a paragraph: did LoRA recover epistemic counts? accuracy? length?

    Works with just {baseline, lora}; pretrained is optional and only used to compute
    a recovery-percentage when present.
    """
    if not rows:
        print("\n(no rows to summarize)")
        return

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    def avg(name, key):
        rs = by_model.get(name, [])
        if not rs:
            return None
        return sum(r[key] for r in rs) / len(rs)

    baseline_label = next((m for m in by_model if "baseline" in m), None)
    lora_label = next((m for m in by_model if "lora" in m), None)
    pretrained_label = next((m for m in by_model if "pretrained" in m), None)

    if not (baseline_label and lora_label):
        print("\nQualitative summary skipped: need baseline and lora rows.")
        return

    n_sampling = rows[0].get("n_sampling", "N")
    acc_key = f"acc@{n_sampling}"

    base_ep = avg(baseline_label, "avg_epistemic_per_response")
    lora_ep = avg(lora_label, "avg_epistemic_per_response")
    pre_ep = avg(pretrained_label, "avg_epistemic_per_response") if pretrained_label else None
    base_acc = avg(baseline_label, acc_key)
    lora_acc = avg(lora_label, acc_key)
    pre_acc = avg(pretrained_label, acc_key) if pretrained_label else None
    base_any = avg(baseline_label, "any_correct_rate")
    lora_any = avg(lora_label, "any_correct_rate")
    pre_any = avg(pretrained_label, "any_correct_rate") if pretrained_label else None
    base_len = avg(baseline_label, "avg_response_length")
    lora_len = avg(lora_label, "avg_response_length")
    pre_len = avg(pretrained_label, "avg_response_length") if pretrained_label else None
    # Standard errors (averaged across benchmarks).
    base_se_any = avg(baseline_label, "se_any_correct_rate") or 0.0
    lora_se_any = avg(lora_label, "se_any_correct_rate") or 0.0
    pre_se_any = (avg(pretrained_label, "se_any_correct_rate") or 0.0) if pretrained_label else None
    base_se_acc = avg(baseline_label, f"se_acc@{n_sampling}") or 0.0
    lora_se_acc = avg(lora_label, f"se_acc@{n_sampling}") or 0.0

    def pct_recovered(base, lora, pre):
        if base is None or lora is None or pre is None:
            return None
        denom = pre - base
        if abs(denom) < 1e-9:
            return None
        return (lora - base) / denom * 100

    ep_recov = pct_recovered(base_ep, lora_ep, pre_ep)
    acc_recov = pct_recovered(base_acc, lora_acc, pre_acc)
    any_recov = pct_recovered(base_any, lora_any, pre_any)

    def fmt_line(label, base, lora, pre, width=22, precision=3):
        line = f"  {label.ljust(width)} base={base:.{precision}f}   lora={lora:.{precision}f}"
        if pre is not None:
            line += f"   pretrained={pre:.{precision}f}"
        return line

    print("\n" + "=" * 70)
    print("QUALITATIVE SUMMARY")
    print("=" * 70)
    print("Headline metric — avg epistemic tokens per response:")
    print(fmt_line("per response:", base_ep, lora_ep, pre_ep, precision=2))
    if ep_recov is not None:
        print(f"  → LoRA recovers {ep_recov:.0f}% of the SDPO suppression toward pretrained.")
    else:
        delta_pct = (lora_ep - base_ep) / base_ep * 100 if base_ep else 0
        print(f"  → LoRA {'+' if delta_pct >= 0 else ''}{delta_pct:.0f}% vs baseline.")

    print("\nAccuracy:")
    print(fmt_line(f"{acc_key} (≈ pass@1):", base_acc, lora_acc, pre_acc))
    print(f"  SE({acc_key}): baseline=±{base_se_acc:.3f}  lora=±{lora_se_acc:.3f}")
    print(fmt_line(f"any-correct (pass@{n_sampling}):", base_any, lora_any, pre_any))
    se_pre_str = f"  pretrained=±{pre_se_any:.3f}" if pre_se_any is not None else ""
    print(f"  SE(any-correct):  baseline=±{base_se_any:.3f}  lora=±{lora_se_any:.3f}{se_pre_str}")
    if acc_recov is not None:
        print(f"  → LoRA recovers {acc_recov:.0f}% of the {acc_key} gap toward pretrained.")
    if any_recov is not None:
        print(f"  → LoRA recovers {any_recov:.0f}% of the any-correct gap toward pretrained.")

    print("\nAvg response length (tokens):")
    print(fmt_line("length:", base_len, lora_len, pre_len, precision=0))
    if pre_len is not None:
        if lora_len < pre_len:
            print(f"  → LoRA stays {pre_len - lora_len:.0f} tokens shorter than pretrained — "
                  f"best-of-both-worlds outcome holds.")
        else:
            print(f"  → LoRA is LONGER than pretrained by {lora_len - pre_len:.0f} tokens — "
                  f"length-control benefit of SDPO has been undone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--csv_out", default="results/epistemic_summary.csv")
    parser.add_argument("--bars_out", default="results/epistemic_comparison.png")
    parser.add_argument("--scatter_out", default="results/length_vs_accuracy.png")
    parser.add_argument("--epistemic_bars_out",
                        default="results/epistemic_per_response.png",
                        help="Headline bar chart: avg epistemic tokens per response, per benchmark.")
    parser.add_argument("--accuracy_bars_out",
                        default="results/accuracy_comparison.png",
                        help="Bar chart of any-correct rate (pass@N) per benchmark.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows, no_think_warnings = aggregate(results_dir)

    print_table(rows)
    write_csv(rows, Path(args.csv_out))
    plot_epistemic_per_response_bars(rows, Path(args.epistemic_bars_out))
    plot_any_correct_bars(rows, Path(args.accuracy_bars_out))
    plot_per_token_bars(rows, Path(args.bars_out))
    plot_length_vs_accuracy(rows, Path(args.scatter_out))

    if no_think_warnings:
        print("\nNote: generations without an explicit </think> closing tag "
              "(usually truncated at max_tokens — counted over full post-<think> span):")
        for label, bench, n, total in no_think_warnings:
            print(f"  {label} / {bench}: {n}/{total} samples")

    qualitative_summary(rows)


if __name__ == "__main__":
    main()
