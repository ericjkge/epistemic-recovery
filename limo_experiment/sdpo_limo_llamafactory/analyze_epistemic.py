#!/usr/bin/env python3
"""Post-hoc epistemic-token analysis over results/ JSON files from evaluate_aime.py.

Counts the Kim et al. 10-token set inside each generation's <think>...</think> span,
case-insensitive, whole-word only. Produces a CSV summary, a per-token bar chart, and
a length-vs-accuracy scatter, then prints a one-paragraph qualitative summary.
"""
import argparse
import csv
import json
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
    """Return (thinking_text, had_explicit_block: bool)."""
    m = THINK_RE.search(text)
    if m:
        return m.group(1), True
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

        # acc@n: mean over all (problem, sample) pairs
        flat_correct = [c for r in results for c in r["correctness"]]
        acc = sum(flat_correct) / len(flat_correct) if flat_correct else 0.0

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

        row = {
            "model": label,
            "benchmark": bench,
            "acc@16": round(acc, 4),
            "avg_response_length": round(avg_resp_len, 1),
            "avg_thinking_length": round(avg_think_len, 1),
            "total_epistemic_tokens_per_response": round(total_epistemic_per_resp, 3),
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
    cols = ["model", "benchmark", "acc@16", "avg_response_length",
            "avg_thinking_length", "total_epistemic_tokens_per_response"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


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


def plot_length_vs_accuracy(rows, out_png: Path):
    benches = sorted({r["benchmark"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = {"aime24": "o", "aime25": "s"}
    for r in rows:
        ax.scatter(r["avg_response_length"], r["acc@16"],
                   marker=markers.get(r["benchmark"], "^"), s=80, label=f"{r['model']} / {r['benchmark']}")
        ax.annotate(r["model"], (r["avg_response_length"], r["acc@16"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("avg response length (tokens)")
    ax.set_ylabel("acc@16")
    ax.set_title("Length vs accuracy across models")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def qualitative_summary(rows):
    """Print a paragraph: did LoRA recover epistemic counts? accuracy? length?"""
    if not rows:
        print("\n(no rows to summarize)")
        return

    # Find canonical labels — we expect baseline / lora / pretrained.
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

    if not (baseline_label and lora_label and pretrained_label):
        print("\nQualitative summary skipped: need baseline / lora / pretrained rows.")
        return

    base_ep = avg(baseline_label, "total_epistemic_tokens_per_response")
    lora_ep = avg(lora_label, "total_epistemic_tokens_per_response")
    pre_ep = avg(pretrained_label, "total_epistemic_tokens_per_response")
    base_acc = avg(baseline_label, "acc@16")
    lora_acc = avg(lora_label, "acc@16")
    pre_acc = avg(pretrained_label, "acc@16")
    base_len = avg(baseline_label, "avg_response_length")
    lora_len = avg(lora_label, "avg_response_length")
    pre_len = avg(pretrained_label, "avg_response_length")

    def pct_recovered(base, lora, pre):
        denom = pre - base
        if abs(denom) < 1e-9:
            return None
        return (lora - base) / denom * 100

    ep_recov = pct_recovered(base_ep, lora_ep, pre_ep)
    acc_recov = pct_recovered(base_acc, lora_acc, pre_acc)

    print("\n" + "=" * 70)
    print("QUALITATIVE SUMMARY")
    print("=" * 70)
    print(
        f"Epistemic tokens / response (mean across benchmarks):\n"
        f"  baseline (SDPO):       {base_ep:.2f}\n"
        f"  + LoRA on LIMO:        {lora_ep:.2f}\n"
        f"  pretrained Qwen3-8B:   {pre_ep:.2f}"
        + (f"\n  → LoRA recovers {ep_recov:.0f}% of the SDPO suppression toward pretrained." if ep_recov is not None else "")
    )
    print(
        f"\nAcc@16 (mean across benchmarks):\n"
        f"  baseline (SDPO):       {base_acc:.3f}\n"
        f"  + LoRA on LIMO:        {lora_acc:.3f}\n"
        f"  pretrained Qwen3-8B:   {pre_acc:.3f}"
        + (f"\n  → LoRA recovers {acc_recov:.0f}% of the accuracy gap toward pretrained." if acc_recov is not None else "")
    )
    print(
        f"\nAvg response length (tokens):\n"
        f"  baseline (SDPO):       {base_len:.0f}\n"
        f"  + LoRA on LIMO:        {lora_len:.0f}\n"
        f"  pretrained Qwen3-8B:   {pre_len:.0f}"
    )
    if lora_len is not None and pre_len is not None:
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
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows, no_think_warnings = aggregate(results_dir)

    print_table(rows)
    write_csv(rows, Path(args.csv_out))
    plot_per_token_bars(rows, Path(args.bars_out))
    plot_length_vs_accuracy(rows, Path(args.scatter_out))

    if no_think_warnings:
        print("\nWARNING: generations without an explicit <think>...</think> block "
              "(counted across whole response):")
        for label, bench, n, total in no_think_warnings:
            print(f"  {label} / {bench}: {n}/{total} samples")

    qualitative_summary(rows)


if __name__ == "__main__":
    main()
