#!/usr/bin/env python3
"""Plot epistemic alignment metrics over training steps from probe JSON snapshots.

Reads the JSON files saved by probe_during_training.py and produces a single PNG
with one subplot per metric, showing how each metric evolves across checkpoints.
Reference baseline values (base_qwen3_8b, sdpo_no_lora) are drawn as horizontal
dashed lines so you can see whether the LoRA is converging toward the target.

Usage:
    python eval/plot_probe_curves.py --probe_dir results/myrun/epistemic_probes
    python eval/plot_probe_curves.py --probe_dir results/myrun/epistemic_probes \
        --output results/myrun/probe_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Metrics to plot, in panel order.
# Each entry: (key in scalars dict, y-axis label, higher-is-better flag)
PANELS = [
    ("epistemic_alignment_gap",   "Alignment gap\n(all − epistemic logprob, ↓ better)", False),
    ("epistemic_entropy_ratio",   "Entropy ratio\n(epistemic / all, ↑ better)",          True),
    ("trace_mean_logprob",        "Trace mean log-prob\n(overall LIMO fit, ↑ better)",   True),
    ("epistemic_mean_logprob",    "Epistemic mean log-prob\n(↑ better)",                 True),
    ("mean_logprob_wait",         "Mean log-prob\n\"wait\"",                              True),
    ("mean_logprob_alternatively","Mean log-prob\n\"alternatively\"",                     True),
    ("mean_logprob_maybe",        "Mean log-prob\n\"maybe\"",                             True),
    ("mean_logprob_hmm",          "Mean log-prob\n\"hmm\"",                              True),
]

REFERENCE_STYLES = {
    "base_qwen3_8b":  dict(color="#2ca02c", linestyle="--", linewidth=1.5, label="base Qwen3-8B (target)"),
    "sdpo_no_lora":   dict(color="#d62728", linestyle=":",  linewidth=1.5, label="SDPO no-LoRA (floor)"),
}

LORA_COLOR = "#1f77b4"


def load_snapshots(probe_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Return (lora_snapshots_sorted_by_step, {ref_label: scalars})."""
    lora = []
    refs = {}

    for path in sorted(probe_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        label = data.get("label", path.stem)
        scalars = data.get("scalars", {})
        step = data.get("step", 0)

        if label in REFERENCE_STYLES:
            refs[label] = scalars
        else:
            lora.append({"step": step, "scalars": scalars, "label": label})

    lora.sort(key=lambda x: x["step"])
    return lora, refs


def plot_curves(lora: list[dict], refs: dict[str, dict], output: Path) -> None:
    n_panels = len(PANELS)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Epistemic alignment metrics over LoRA training", fontsize=13, fontweight="bold", y=1.01)

    steps = [s["step"] for s in lora]

    for idx, (key, ylabel, higher_better) in enumerate(PANELS):
        ax = axes[idx // ncols][idx % ncols]

        # LoRA curve
        values = [s["scalars"].get(key, float("nan")) for s in lora]
        valid = [(s, v) for s, v in zip(steps, values) if not (v != v)]  # drop NaN
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color=LORA_COLOR, linewidth=2, marker="o", markersize=5, label="LoRA (this run)")

        # Reference baselines
        for ref_label, ref_scalars in refs.items():
            val = ref_scalars.get(key, float("nan"))
            if val != val:
                continue
            style = REFERENCE_STYLES[ref_label]
            ax.axhline(val, **style)

        ax.set_xlabel("Training step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(key, fontsize=8, color="#444")
        ax.grid(True, alpha=0.3)

        if steps:
            ax.set_xlim(left=0)

        # Arrow in corner showing which direction is better
        direction = "↑" if higher_better else "↓"
        ax.text(0.98, 0.04, direction, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=12, color="#888")

    # Hide unused panels
    for idx in range(len(PANELS), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=9, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output}")


def print_summary(lora: list[dict], refs: dict[str, dict]) -> None:
    if not lora:
        print("No LoRA checkpoints found.")
        return

    print(f"\n{'step':>6}  {'gap':>8}  {'ent_ratio':>9}  {'trace_lp':>8}  {'ep_lp':>8}")
    print("-" * 48)
    for s in lora:
        sc = s["scalars"]
        print(
            f"{s['step']:>6}  "
            f"{sc.get('epistemic_alignment_gap', float('nan')):>8.4f}  "
            f"{sc.get('epistemic_entropy_ratio', float('nan')):>9.4f}  "
            f"{sc.get('trace_mean_logprob', float('nan')):>8.4f}  "
            f"{sc.get('epistemic_mean_logprob', float('nan')):>8.4f}"
        )

    if refs:
        print()
        for ref_label, sc in refs.items():
            print(
                f"{'[' + ref_label + ']':>30}  "
                f"gap={sc.get('epistemic_alignment_gap', float('nan')):.4f}  "
                f"ent_ratio={sc.get('epistemic_entropy_ratio', float('nan')):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot epistemic probe metrics over training steps.")
    parser.add_argument("--probe_dir", required=True,
                        help="Directory containing probe JSON snapshots (results/RUN_ID/epistemic_probes).")
    parser.add_argument("--output", default=None,
                        help="Output PNG path. Defaults to probe_dir/../probe_curves.png.")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    if not probe_dir.exists():
        print(f"ERROR: probe_dir not found: {probe_dir}")
        raise SystemExit(1)

    output = Path(args.output) if args.output else probe_dir.parent / "probe_curves.png"

    lora, refs = load_snapshots(probe_dir)
    print(f"Loaded {len(lora)} LoRA checkpoint(s) and {len(refs)} reference baseline(s).")
    print_summary(lora, refs)
    plot_curves(lora, refs, output)


if __name__ == "__main__":
    main()
