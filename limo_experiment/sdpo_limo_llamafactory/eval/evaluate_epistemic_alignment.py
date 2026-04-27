#!/usr/bin/env python3
"""Figure 9 reproduction: epistemic alignment diagnostic plots for cross-model comparison.

Implements the three diagnostic plot types from Kim et al. (arXiv:2603.15500):
  Plot A — CDF of per-trace mean log-probability (all models overlaid)
  Plot B — token log-prob density per model (all tokens vs. "wait" vs. "alternatively")
  Plot C — token entropy density per model (same structure as B)

Layout: CDF on page 1; B + C grid (rows = models, cols = logprob/entropy) on page 2.

Two modes:
  --from_json   Load pre-computed JSON snapshots from probe_during_training.py (fast).
  --models      Compute metrics fresh from model checkpoints (slow, requires GPU).

Usage:
    # From pre-computed probe snapshots (recommended after a training run):
    python eval/evaluate_epistemic_alignment.py \\
        --from_json \\
            results/my_run/epistemic_probes/base_qwen3_8b_step0000000.json \\
            results/my_run/epistemic_probes/sdpo_no_lora_step0000000.json \\
            results/my_run/epistemic_probes/lora_step0007000.json \\
        --output results/my_run/figure9.pdf

    # Compute fresh from model paths (format: "label:model_path" or "label:base+adapter"):
    python eval/evaluate_epistemic_alignment.py \\
        --probe_dataset limo_v2_sdpo.json \\
        --models \\
            "pretrained:Qwen/Qwen3-8B" \\
            "sdpo:beanie00/math-SDPO-Qwen3-8B-think-step-100" \\
            "lora:beanie00/math-SDPO-Qwen3-8B-think-step-100+outputs/my_run" \\
        --output results/my_run/figure9.pdf

Run from the sdpo_limo_llamafactory/ parent directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── KDE helper ────────────────────────────────────────────────────────────────

_MIN_KDE_POINTS = 5


def _kde(values: list, x_grid: np.ndarray) -> Optional[np.ndarray]:
    """Evaluate a KDE on x_grid; returns None when there are too few points."""
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < _MIN_KDE_POINTS:
        return None
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(arr, bw_method="scott")(x_grid)
    except Exception:
        # Fallback to histogram-based density estimate
        counts, edges = np.histogram(arr, bins=len(x_grid), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        return np.interp(x_grid, centers, counts, left=0.0, right=0.0)


# ── Color / style constants ───────────────────────────────────────────────────

_TEAL = "#2ca0a0"
_RED = "#d62728"
_BLUE = "#1f77b4"
_MODEL_PALETTE = plt.cm.tab10.colors


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_from_json(path: str) -> dict:
    """Load pre-computed metrics from a probe_during_training.py JSON snapshot."""
    with open(path, "r") as f:
        data = json.load(f)
    merged = {"label": data["label"]}
    merged.update(data.get("scalars", {}))
    merged.update(data.get("arrays", {}))
    return merged


def _compute_from_model(
    model_path: str,
    label: str,
    adapter_path: Optional[str],
    probe_dataset: str,
    n_probes: int,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from epistemic_metrics import (
        build_diagnostic_token_map,
        build_epistemic_token_ids,
        compute_epistemic_metrics,
        load_probe_set,
        prepare_probe_set,
    )

    print(f"\nLoading {label}: {model_path}" + (f" + {adapter_path}" if adapter_path else ""), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    ep_ids = build_epistemic_token_ids(tokenizer)
    diag_map = build_diagnostic_token_map(tokenizer)
    probe_traces = load_probe_set(probe_dataset, n_probes=n_probes)
    prepared = prepare_probe_set(tokenizer, probe_traces)
    print(f"  probing {len(prepared)} traces...", flush=True)

    metrics = compute_epistemic_metrics(model, tokenizer, probe_traces, ep_ids, diag_map, prepared=prepared)
    metrics["label"] = label

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


# ── Plot A: CDF ───────────────────────────────────────────────────────────────

def _plot_cdf(ax, all_metrics: list[dict]) -> None:
    for i, m in enumerate(all_metrics):
        vals = m.get("per_trace_avg_logprobs", [])
        if not vals:
            continue
        arr = np.sort(vals)
        cdf = np.arange(1, len(arr) + 1) / len(arr) * 100
        ax.plot(arr, cdf, label=m["label"], color=_MODEL_PALETTE[i % len(_MODEL_PALETTE)], linewidth=2)
    ax.set_xlabel("Avg log-prob per trace", fontsize=11)
    ax.set_ylabel("Cumulative probability (%)", fontsize=11)
    ax.set_title("CDF of per-trace mean log-probability", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ── Plot B: log-prob density (per model) ─────────────────────────────────────

def _plot_logprob_density(ax, metrics: dict, xlim: tuple = (-15, 0)) -> None:
    x_grid = np.linspace(xlim[0], xlim[1], 500)

    y_all = _kde(metrics.get("all_token_logprobs_sample", []), x_grid)
    if y_all is not None:
        ax.plot(x_grid, y_all, color=_TEAL, linewidth=2, label="All tokens")

    wait_lp = metrics.get("wait_token_logprobs", [])
    y_wait = _kde(wait_lp, x_grid)
    if y_wait is not None:
        ax.plot(x_grid, y_wait, color=_RED, linewidth=1.5, linestyle="--", label='"wait"')
        ax.axvline(float(np.mean(wait_lp)), color=_RED, linestyle=":", linewidth=1.0, alpha=0.75)
    elif wait_lp:
        ax.axvline(float(np.mean(wait_lp)), color=_RED, linestyle=":", linewidth=1.5,
                   label=f'"wait" mean (n={len(wait_lp)})')

    alt_lp = metrics.get("alternatively_token_logprobs", [])
    y_alt = _kde(alt_lp, x_grid)
    if y_alt is not None:
        ax.plot(x_grid, y_alt, color=_BLUE, linewidth=1.5, linestyle="--", label='"alternatively"')
        ax.axvline(float(np.mean(alt_lp)), color=_BLUE, linestyle=":", linewidth=1.0, alpha=0.75)
    elif alt_lp:
        ax.axvline(float(np.mean(alt_lp)), color=_BLUE, linestyle=":", linewidth=1.5,
                   label=f'"alternatively" mean (n={len(alt_lp)})')

    ax.set_xlim(xlim)
    ax.set_xlabel("Log-probability", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{metrics['label']} — log-prob density", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


# ── Plot C: entropy density (per model) ──────────────────────────────────────

def _plot_entropy_density(ax, metrics: dict, xlim: tuple = (0, 5)) -> None:
    x_grid = np.linspace(xlim[0], xlim[1], 500)

    y_all = _kde(metrics.get("all_token_entropies_sample", []), x_grid)
    if y_all is not None:
        ax.plot(x_grid, y_all, color=_TEAL, linewidth=2, label="All tokens")

    wait_ent = metrics.get("wait_token_entropies", [])
    y_wait = _kde(wait_ent, x_grid)
    if y_wait is not None:
        ax.plot(x_grid, y_wait, color=_RED, linewidth=1.5, linestyle="--", label='"wait"')
        ax.axvline(float(np.mean(wait_ent)), color=_RED, linestyle=":", linewidth=1.0, alpha=0.75)
    elif wait_ent:
        ax.axvline(float(np.mean(wait_ent)), color=_RED, linestyle=":", linewidth=1.5,
                   label=f'"wait" mean (n={len(wait_ent)})')

    alt_ent = metrics.get("alternatively_token_entropies", [])
    y_alt = _kde(alt_ent, x_grid)
    if y_alt is not None:
        ax.plot(x_grid, y_alt, color=_BLUE, linewidth=1.5, linestyle="--", label='"alternatively"')
        ax.axvline(float(np.mean(alt_ent)), color=_BLUE, linestyle=":", linewidth=1.0, alpha=0.75)
    elif alt_ent:
        ax.axvline(float(np.mean(alt_ent)), color=_BLUE, linestyle=":", linewidth=1.5,
                   label=f'"alternatively" mean (n={len(alt_ent)})')

    ax.set_xlim(xlim)
    ax.set_xlabel("Entropy", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{metrics['label']} — entropy density", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


# ── Figure assembly ───────────────────────────────────────────────────────────

def make_figure9(all_metrics: list[dict], output_path: str) -> None:
    """Write a two-page PDF:
    Page 1 — CDF plot (Plot A).
    Page 2 — n_models × 2 grid of log-prob and entropy densities (Plots B & C).
    """
    n_models = len(all_metrics)

    with PdfPages(output_path) as pdf:
        # ── Page 1: CDF ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_cdf(ax, all_metrics)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: density grid ──────────────────────────────────────────────
        fig, axes = plt.subplots(
            n_models, 2,
            figsize=(12, 4 * n_models),
            squeeze=False,
        )
        for row, m in enumerate(all_metrics):
            _plot_logprob_density(axes[row, 0], m)
            _plot_entropy_density(axes[row, 1], m)

        fig.suptitle(
            "Figure 9 — log-prob & entropy density per model\n"
            "(teal: all tokens | red dashed: \"wait\" | blue dashed: \"alternatively\")",
            fontsize=11, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved Figure 9 → {output_path}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_model_spec(spec: str) -> tuple[str, str, Optional[str]]:
    """Parse 'label:model_path' or 'label:base_path+adapter_path'."""
    label, rest = spec.split(":", 1)
    if "+" in rest:
        base, adapter = rest.split("+", 1)
        return label.strip(), base.strip(), adapter.strip()
    return label.strip(), rest.strip(), None


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 9 epistemic alignment diagnostic plots.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--models", nargs="+",
        metavar="LABEL:MODEL[+ADAPTER]",
        help="Compute metrics fresh. Format: 'label:path' or 'label:base+adapter_path'.",
    )
    source.add_argument(
        "--from_json", nargs="+",
        metavar="JSON_FILE",
        help="Load pre-computed metrics JSON files (from probe_during_training.py).",
    )

    parser.add_argument("--probe_dataset", default="limo_v2_sdpo.json",
                        help="LIMO-format JSON for probe traces (used with --models).")
    parser.add_argument("--n_probes", type=int, default=50)
    parser.add_argument("--output", default="results/figure9.pdf")
    parser.add_argument("--logprob_xlim", nargs=2, type=float, default=[-15, 0],
                        metavar=("LO", "HI"), help="X-axis range for log-prob density plots.")
    parser.add_argument("--entropy_xlim", nargs=2, type=float, default=[0, 5],
                        metavar=("LO", "HI"), help="X-axis range for entropy density plots.")
    args = parser.parse_args()

    all_metrics: list[dict] = []

    if args.from_json:
        for path in args.from_json:
            print(f"Loading pre-computed metrics: {path}", flush=True)
            all_metrics.append(_load_from_json(path))
    else:
        for spec in args.models:
            label, model_path, adapter_path = _parse_model_spec(spec)
            m = _compute_from_model(model_path, label, adapter_path, args.probe_dataset, args.n_probes)
            all_metrics.append(m)

    if not all_metrics:
        print("No metrics loaded. Exiting.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Monkey-patch plot functions to honour xlim args
    import functools
    global _plot_logprob_density, _plot_entropy_density
    _orig_lp = _plot_logprob_density
    _orig_ent = _plot_entropy_density
    _plot_logprob_density = functools.partial(_orig_lp, xlim=tuple(args.logprob_xlim))
    _plot_entropy_density = functools.partial(_orig_ent, xlim=tuple(args.entropy_xlim))

    make_figure9(all_metrics, args.output)

    # Print scalar summary table
    print("\nScalar summary:")
    cols = ["label", "epistemic_alignment_gap", "epistemic_entropy_ratio",
            "trace_mean_logprob", "epistemic_mean_logprob"]
    header = "  ".join(f"{c:<28}" for c in cols)
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        row = "  ".join(
            f"{str(m.get(c, 'N/A')):<28}" if c == "label"
            else f"{m.get(c, float('nan')):<28.4f}"
            for c in cols
        )
        print(row)


if __name__ == "__main__":
    main()
