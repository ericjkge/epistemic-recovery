#!/usr/bin/env python3
"""Aggregate experiment4 sweep cells into a ranked summary.

For each cell directory under results/, pulls:
  - hparams from cell.json
  - latest LoRA probe snapshot (highest step) from eval_results/epistemic_probes/
  - SDPO floor and pretrained target from the same probe directory (or fall back to
    experiment2/eval_results/epistemic_probes/ since the references don't change
    between runs)
  - AIME pass@k + epistemic counts from eval_results/epistemic_summary.csv
  - loop rate from eval_results/epistemic_summary.csv if present

Computes:
  recovery_pct = (lora_epistemic_logprob - sdpo) / (pretrained - sdpo)
                 1.0 == matched pretrained; <0.5 under-recovery; >1.2 over-mimicry

Outputs:
  summary.csv               one row per cell, sorted by |recovery_pct - 1|
  recovery_vs_passk.png     scatter of recovery_pct (x) vs AIME pass@k (y)
  recovery_distribution.png bar chart of recovery_pct per cell with target band

Usage:
    python3 analyze_sweep.py --results_dir results --output summary.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP4_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP4_DIR.parent.parent  # sdpo_limo_llamafactory/

# Reference probe locations to fall back on if the cell didn't recompute floor/target.
FALLBACK_PROBE_DIRS = [
    PARENT_DIR / "experiments/experiment2/eval_results/epistemic_probes",
    PARENT_DIR / "experiments/experiment1/eval_results/epistemic_probes",
]


def _latest_step_snapshot(probe_dir: Path, prefix: str) -> dict | None:
    """Return the highest-step probe snapshot matching `{prefix}_step*.json`, or None."""
    candidates = sorted(probe_dir.glob(f"{prefix}_step*.json"))
    if not candidates:
        return None
    # filename pattern: {prefix}_step{N:07d}.json
    def step_of(p: Path) -> int:
        m = re.search(r"_step(\d+)", p.stem)
        return int(m.group(1)) if m else 0
    chosen = max(candidates, key=step_of)
    with open(chosen) as f:
        d = json.load(f)
    d["_path"] = str(chosen)
    return d


def _resolve_reference(probe_dir: Path, prefix: str) -> dict | None:
    """Search probe_dir then FALLBACK_PROBE_DIRS for a snapshot."""
    for root in [probe_dir, *FALLBACK_PROBE_DIRS]:
        if not root.exists():
            continue
        snap = _latest_step_snapshot(root, prefix)
        if snap is not None:
            return snap
    return None


def _read_epistemic_summary(csv_path: Path) -> dict:
    """Parse epistemic_summary.csv. Returns {(model, benchmark): row} dict."""
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[(row.get("model", ""), row.get("benchmark", ""))] = row
    return out


def _first_acc_key(row: dict) -> tuple[str | None, float | None]:
    """The acc column is `acc@N` where N varies — return (key, value)."""
    for k, v in row.items():
        if k.startswith("acc@"):
            try:
                return k, float(v)
            except (TypeError, ValueError):
                return k, None
    return None, None


def collect_cell(cell_dir: Path) -> dict | None:
    cell_json = cell_dir / "cell.json"
    if not cell_json.exists():
        return None
    cell = json.loads(cell_json.read_text())
    probe_dir = cell_dir / "eval_results" / "epistemic_probes"

    sdpo = _resolve_reference(probe_dir, "sdpo_no_lora")
    base = _resolve_reference(probe_dir, "base_qwen3_8b")
    lora = _latest_step_snapshot(probe_dir, "lora")

    # Numeric scalars or None.
    def s(d, key):
        if d is None:
            return None
        return d.get("scalars", {}).get(key)

    e_lora = s(lora, "epistemic_mean_logprob")
    e_sdpo = s(sdpo, "epistemic_mean_logprob")
    e_base = s(base, "epistemic_mean_logprob")
    recovery = None
    if None not in (e_lora, e_sdpo, e_base) and e_base != e_sdpo:
        recovery = (e_lora - e_sdpo) / (e_base - e_sdpo)

    # AIME numbers — read whatever benchmark/model rows exist.
    summary = _read_epistemic_summary(cell_dir / "eval_results" / "epistemic_summary.csv")
    lora_rows = [(k, v) for k, v in summary.items() if "lora" in k[0].lower()]
    aime_metrics: dict = {}
    for (model, bench), row in lora_rows:
        acc_key, acc = _first_acc_key(row)
        aime_metrics[bench] = {
            "model": model,
            "acc_key": acc_key,
            "acc": acc,
            "any_correct": float(row["any_correct_rate"]) if row.get("any_correct_rate") else None,
            "avg_response_length": float(row["avg_response_length"]) if row.get("avg_response_length") else None,
            "avg_epistemic_per_response": (
                float(row["avg_epistemic_per_response"])
                if row.get("avg_epistemic_per_response") else None
            ),
        }

    return {
        "cell": cell["cell"],
        "rank": cell["rank"],
        "epochs": cell["epochs"],
        "lr": cell["lr"],
        "lora_step": (lora or {}).get("step"),
        "epistemic_logprob_lora": e_lora,
        "epistemic_logprob_sdpo": e_sdpo,
        "epistemic_logprob_pretrained": e_base,
        "all_logprob_lora": s(lora, "all_tokens_mean_logprob"),
        "alignment_gap_lora": s(lora, "epistemic_alignment_gap"),
        "recovery_pct": recovery,
        "aime": aime_metrics,
    }


def write_summary_csv(rows: list[dict], path: Path):
    benches = sorted({b for r in rows for b in r["aime"].keys()})
    fieldnames = [
        "cell", "rank", "epochs", "lr", "lora_step",
        "epistemic_logprob_lora", "epistemic_logprob_sdpo", "epistemic_logprob_pretrained",
        "alignment_gap_lora", "all_logprob_lora", "recovery_pct",
    ]
    for b in benches:
        for col in ("acc", "any_correct", "avg_response_length", "avg_epistemic_per_response"):
            fieldnames.append(f"{b}_{col}")

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames if k in r}
            for b in benches:
                m = r["aime"].get(b, {})
                row[f"{b}_acc"] = m.get("acc")
                row[f"{b}_any_correct"] = m.get("any_correct")
                row[f"{b}_avg_response_length"] = m.get("avg_response_length")
                row[f"{b}_avg_epistemic_per_response"] = m.get("avg_epistemic_per_response")
            w.writerow(row)


def plot_recovery(rows: list[dict], out_dir: Path):
    rows_with_rec = [r for r in rows if r.get("recovery_pct") is not None]
    if not rows_with_rec:
        return
    rows_with_rec.sort(key=lambda r: r["recovery_pct"])

    # Bar chart of recovery_pct with target band.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r["cell"] for r in rows_with_rec]
    recs = [r["recovery_pct"] for r in rows_with_rec]
    bars = ax.bar(range(len(recs)), recs, color="#2ca0a0")
    for i, r in enumerate(recs):
        if r > 1.2:
            bars[i].set_color("#d62728")  # over-mimicry
        elif r < 0.5:
            bars[i].set_color("#bbbbbb")  # under-recovery
    ax.axhspan(0.7, 1.1, color="#9bd99b", alpha=0.3, label="target band [0.7, 1.1]")
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1, label="pretrained target")
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1, label="sdpo floor")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("recovery_pct")
    ax.set_title("Epistemic recovery vs pretrained target (1.0 = match)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_distribution.png", dpi=150)
    plt.close(fig)

    # Scatter: recovery_pct vs AIME pass@k (any_correct).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plotted = 0
    for r in rows_with_rec:
        for b, m in r["aime"].items():
            if m.get("any_correct") is None:
                continue
            ax.scatter(r["recovery_pct"], m["any_correct"], s=60, alpha=0.85)
            ax.annotate(
                f"{r['cell']}\n{b}",
                (r["recovery_pct"], m["any_correct"]),
                textcoords="offset points", xytext=(5, 4), fontsize=7,
            )
            plotted += 1
    ax.axvspan(0.7, 1.1, color="#9bd99b", alpha=0.3)
    ax.axvline(1.0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("recovery_pct (epistemic logprob)")
    ax.set_ylabel("AIME any-correct rate")
    ax.set_title("Capability vs alignment recovery")
    fig.tight_layout()
    if plotted:
        fig.savefig(out_dir / "recovery_vs_passk.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=str(EXP4_DIR / "results"))
    ap.add_argument("--output", default=str(EXP4_DIR / "summary.csv"))
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    out_csv = Path(args.output)

    rows = []
    for cell_dir in sorted(results_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        row = collect_cell(cell_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"No completed cells under {results_root}.")
        return

    # Sort by |recovery_pct - 1|, missing recovery sinks to the bottom.
    def sort_key(r):
        rec = r.get("recovery_pct")
        return (rec is None, abs(rec - 1.0) if rec is not None else 99.0)
    rows.sort(key=sort_key)

    write_summary_csv(rows, out_csv)
    plot_recovery(rows, out_csv.parent)

    # Write best_config.json with the top-ranked cell that has a recovery_pct
    # in the [0.7, 1.1] target band; falls back to the closest-to-1.0 cell. This
    # is what experiment 5's train.sh consumes.
    with_rec = [r for r in rows if r.get("recovery_pct") is not None]
    in_band = [r for r in with_rec if 0.7 <= r["recovery_pct"] <= 1.1]
    pick = (in_band or with_rec or [None])[0]
    if pick is not None:
        best_path = out_csv.parent / "best_config.json"
        # Tie-break inside the band: prefer the cell with the highest pass rate.
        if in_band:
            def best_acc(r):
                accs = [m.get("any_correct") for m in r["aime"].values() if m.get("any_correct") is not None]
                return max(accs) if accs else -1
            pick = max(in_band, key=best_acc)
        with open(best_path, "w") as f:
            json.dump({
                "cell": pick["cell"],
                "rank": pick["rank"],
                "epochs": pick["epochs"],
                "lr": pick["lr"],
                "recovery_pct": pick["recovery_pct"],
                "in_target_band": 0.7 <= pick["recovery_pct"] <= 1.1,
            }, f, indent=2)
        print(f"best_config → {best_path}  (cell={pick['cell']}, recovery={pick['recovery_pct']:+.2f})")

    print(f"\n{'cell':<22} {'rank':>4} {'epochs':>6} {'recovery':>10} {'best_acc':>9}")
    print("─" * 60)
    for r in rows:
        rec = r.get("recovery_pct")
        rec_str = f"{rec:+.2f}" if rec is not None else "  n/a"
        accs = [m["any_correct"] for m in r["aime"].values() if m.get("any_correct") is not None]
        best = max(accs) if accs else None
        best_str = f"{best:.3f}" if best is not None else "  n/a"
        print(f"{r['cell']:<22} {r['rank']:>4} {r['epochs']:>6} {rec_str:>10} {best_str:>9}")
    print(f"\nsummary.csv → {out_csv}")
    print(f"plots       → {out_csv.parent}/recovery_distribution.png, recovery_vs_passk.png")


if __name__ == "__main__":
    main()
