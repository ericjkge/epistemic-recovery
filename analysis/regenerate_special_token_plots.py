"""Regenerate special token distribution plots with model labels in titles.

Reads existing histogram CSVs and special_tokens.csv files in analysis/results/
and re-plots the special_token_dist.png with the model label in the title.
Also prints the average log prob (weighted from the histogram) for each model.
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODELS = [
    {
        "prefix": "Qwen-Qwen3-8B_think-on",
        "label": "Qwen3-8B Base Think",
    },
    {
        "prefix": "beanie00-math-SDPO-Qwen3-8B-think-step-100",
        "label": "SDPO Qwen3-8B Think",
    },
]


def read_histogram_csv(path):
    bin_centers, counts, densities = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = float(row["bin_start"])
            be = float(row["bin_end"])
            bin_centers.append((bs + be) / 2.0)
            counts.append(int(row["count"]))
            densities.append(float(row["density"]))
    return np.array(bin_centers), np.array(counts), np.array(densities)


def read_token_stats_means(path):
    total = 0
    lp_sum = 0.0
    en_sum = 0.0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = int(row["count"])
            total += c
            lp_sum += c * float(row["avg_logprob"])
            en_sum += c * float(row["avg_entropy"])
    return lp_sum / total, en_sum / total, total


def read_special_tokens_csv(path):
    by_token = defaultdict(lambda: {"logprobs": [], "entropies": [], "token": None})
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["token_id"]
            by_token[tid]["token"] = row["token"]
            by_token[tid]["logprobs"].append(float(row["logprob"]))
            by_token[tid]["entropies"].append(float(row["entropy"]))
    return by_token


def weighted_mean(centers, counts):
    total = counts.sum()
    if total == 0:
        return float("nan")
    return float((centers * counts).sum() / total)


def plot_density_from_hist(ax, centers, densities, label, color, line_style="-", mean_value=None):
    ax.plot(centers, densities, label=label, color=color, linestyle=line_style, linewidth=2)
    if mean_value is not None:
        ax.axvline(mean_value, color=color, linestyle=":", linewidth=2, alpha=0.9)


def plot_density_from_values(ax, values, label, color, line_style="--", bins=120):
    if len(values) < 2:
        return
    arr = np.asarray(values, dtype=np.float32)
    hist, edges = np.histogram(arr, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(centers, hist, label=label, color=color, linestyle=line_style, linewidth=2)
    ax.axvline(arr.mean(), color=color, linestyle=":", linewidth=2, alpha=0.9)


SPECIAL_COLORS = [
    "#e53935", "#1e40ff", "#7e57c2", "#f59e0b", "#8d6e63",
    "#00897b", "#f4511e", "#3949ab", "#6d4c41", "#039be5",
]


def regenerate(model):
    prefix = model["prefix"]
    label = model["label"]

    lp_centers, lp_counts, lp_densities = read_histogram_csv(
        os.path.join(RESULTS_DIR, f"{prefix}_all_logprobs.csv")
    )
    en_centers, en_counts, en_densities = read_histogram_csv(
        os.path.join(RESULTS_DIR, f"{prefix}_all_entropies.csv")
    )
    special = read_special_tokens_csv(
        os.path.join(RESULTS_DIR, f"{prefix}_special_tokens.csv")
    )

    avg_lp, avg_en, total_tok = read_token_stats_means(
        os.path.join(RESULTS_DIR, f"{prefix}_token_stats.csv")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_density_from_hist(
        axes[0], lp_centers, lp_densities, "All tokens", "#62c7a5", "-", mean_value=avg_lp,
    )
    plot_density_from_hist(
        axes[1], en_centers, en_densities, "All tokens", "#62c7a5", "-", mean_value=avg_en,
    )

    sorted_items = sorted(
        special.items(), key=lambda kv: len(kv[1]["logprobs"]), reverse=True,
    )
    for idx, (_tid, data) in enumerate(sorted_items):
        color = SPECIAL_COLORS[idx % len(SPECIAL_COLORS)]
        tok = data["token"]
        plot_density_from_values(axes[0], data["logprobs"], tok, color, "--")
        plot_density_from_values(axes[1], data["entropies"], tok, color, "--")

    axes[0].set_title(f"Token Log Probability Density ({label})")
    axes[0].set_xlabel("Log Probability")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"Token Entropy Density ({label})")
    axes[1].set_xlabel("Entropy")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, f"{prefix}_special_token_dist.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{label}] saved → {output_path}")
    print(f"    avg log prob (all tokens): {avg_lp:.4f}   "
          f"avg entropy (all tokens): {avg_en:.4f}   "
          f"n={total_tok:,}")


def main():
    for m in MODELS:
        regenerate(m)


if __name__ == "__main__":
    main()
