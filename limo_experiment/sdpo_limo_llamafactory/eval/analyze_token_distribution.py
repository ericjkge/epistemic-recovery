#!/usr/bin/env python3
"""Token-level log probability and entropy analysis for epistemic verbalization recovery.

Computes teacher-forced per-token log probabilities and Shannon entropy over a dataset,
with special tracking for all epistemic verbalization tokens used in the Kim et al.
10-token set. Designed for pre/post LoRA comparison on the SDPO LIMO pipeline.

Key insight: higher log-prob + lower entropy at epistemic token positions post-LoRA means
the model has learned to assign more decisive probability mass to epistemic verbalizations.

Run from the sdpo_limo_llamafactory/ parent directory:
    source .venv-train/bin/activate

    # Pre-LoRA baseline
    python eval/analyze_token_distribution.py \\
        --model_name_or_path beanie00/math-SDPO-Qwen3-8B-think-step-100 \\
        --input_json data/limo_qwen3_thinking.json \\
        --tag pre_lora

    # Post-LoRA (pass merged checkpoint or adapter via --adapter_path)
    python eval/analyze_token_distribution.py \\
        --model_name_or_path saves/qwen3_sdpo_limo_lora/merged \\
        --input_json data/limo_qwen3_thinking.json \\
        --tag post_lora

    # Compare using outputs from evaluate_aime.py (measures actual generation behavior)
    python eval/analyze_token_distribution.py \\
        --results_json results/lora_sdpo_plus_limo_aime25.json \\
        --model_name_or_path beanie00/math-SDPO-Qwen3-8B-think-step-100 \\
        --adapter_path saves/qwen3_sdpo_limo_lora \\
        --tag post_lora_aime25

Outputs (all under results/token_dist/{tag}/):
    {tag}_token_stats.csv         per-type aggregates: count, avg logprob, avg entropy
    {tag}_special_tokens.csv      per-occurrence logprob/entropy for epistemic tokens
    {tag}_all_logprobs.csv        full logprob histogram (for overlaying distributions)
    {tag}_all_entropies.csv       full entropy histogram
    {tag}_logprob_dist.png        histogram + CDF + box plot + summary stats
    {tag}_special_token_dist.png  logprob/entropy density overlays for epistemic tokens
"""

import json
import os
import argparse
import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# The full Kim et al. epistemic verbalization token set, including capitalized variants
# that appear at sentence/thought boundaries in reasoning chains.
# The script resolves these to token IDs dynamically and skips any that are multi-token.
EPISTEMIC_TOKEN_STRINGS = [
    "wait", "Wait",
    "hmm", "Hmm",
    "perhaps", "Perhaps",
    "maybe", "Maybe",
    "actually", "Actually",
    "alternatively", "Alternatively",
    "seems", "Seems",
    "might", "Might",
    "likely", "Likely",
    "check", "Check",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze token-level log probability and entropy for epistemic verbalization recovery."
    )
    parser.add_argument("--model_name_or_path", type=str,
                        default="beanie00/math-SDPO-Qwen3-8B-think-step-100")
    parser.add_argument("--adapter_path", type=str, default="",
                        help="Path to LoRA adapter directory (PEFT). Leave empty for base/merged models.")
    parser.add_argument("--input_json", type=str, default="data/limo_qwen3_thinking.json",
                        help="Local JSON dataset. Overridden by --results_json if provided.")
    parser.add_argument("--results_json", type=str, default="",
                        help="Path to a results JSON from evaluate_aime.py. When set, analyzes "
                             "the model's own generated outputs instead of gold references.")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="HuggingFace dataset name (alternative to --input_json).")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If > 0, analyze only the first N samples.")
    parser.add_argument("--tag", type=str, default="model",
                        help="Short label used in output filenames (e.g. pre_lora, post_lora).")
    parser.add_argument("--output_dir", type=str, default="results/token_dist",
                        help="Directory for all outputs. Files are prefixed with --tag.")
    parser.add_argument("--special_tokens", type=str, nargs="*",
                        default=EPISTEMIC_TOKEN_STRINGS,
                        help="Token strings to track individually. Defaults to all epistemic "
                             "verbalization tokens (both lower and Title case).")
    parser.add_argument(
        "--dtype", default="bfloat16", type=str,
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument("--question_keys", type=str, nargs="*",
                        default=["instruction", "question", "problem", "prompt"])
    parser.add_argument("--output_keys", type=str, nargs="*",
                        default=["output", "solution", "reasoning", "completion", "response"])
    parser.add_argument("--system_keys", type=str, nargs="*", default=["system"])
    parser.add_argument("--input_keys", type=str, nargs="*", default=["input", "context"])
    return parser.parse_args()


def load_model(model_name_or_path, adapter_path="", dtype="bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype.lower()]

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("--adapter_path requires peft. Install with: pip install peft")
        print(f"  Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def resolve_special_token_ids(tokenizer, token_strings):
    """Resolve token strings to their IDs. Deduplicates by ID (last string wins).

    Returns dict: token_id (int) -> {"token": str, "logprobs": [], "entropies": []}.
    Tokens that encode to multiple sub-tokens are skipped with a warning.
    """
    special_tokens = {}
    for token_str in token_strings:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1:
            if ids[0] in special_tokens:
                existing = special_tokens[ids[0]]["token"]
                print(f"  Note: '{token_str}' maps to same ID {ids[0]} as '{existing}' — merging under '{existing}'")
            else:
                special_tokens[ids[0]] = {"token": token_str, "logprobs": [], "entropies": []}
                print(f"  Tracking: '{token_str}' → ID {ids[0]}")
        else:
            print(f"  WARNING: '{token_str}' encodes to {len(ids)} sub-tokens {ids}, skipping")
    return special_tokens


def build_chat_prompt(tokenizer, system, instruction, user_input):
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    user_content = instruction
    if user_input and user_input.strip():
        user_content = instruction + "\n" + user_input
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def first_present_value(example, keys):
    for k in keys:
        if k in example and example[k] is not None:
            val = str(example[k]).strip()
            if val:
                return val
    return ""


def normalize_example(example, args):
    return {
        "system": first_present_value(example, args.system_keys),
        "instruction": first_present_value(example, args.question_keys),
        "input": first_present_value(example, args.input_keys),
        "output": first_present_value(example, args.output_keys),
    }


def load_examples(args):
    """Load examples from evaluate_aime.py results, local JSON, or HuggingFace."""
    # Priority 1: evaluate_aime.py output — analyze actual generated responses
    if args.results_json:
        print(f"Loading generated outputs from evaluate_aime.py results: {args.results_json}")
        with open(args.results_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data_list = []
        for r in payload["results"]:
            for gen in r["generations"]:
                data_list.append({"question": r["question"], "output": gen})
        print(f"  Loaded {len(data_list)} generation(s) from {len(payload['results'])} problem(s)")
        if args.max_samples > 0:
            data_list = data_list[:args.max_samples]
        return data_list

    # Priority 2: HuggingFace dataset
    if args.dataset_name:
        if load_dataset is None:
            raise ImportError("datasets is required for --dataset_name.")
        print(f"Loading from HuggingFace: {args.dataset_name} [{args.dataset_split}]")
        ds = load_dataset(args.dataset_name, split=args.dataset_split)
        data_list = [dict(x) for x in ds]
    else:
        print(f"Loading from JSON: {args.input_json}")
        with open(args.input_json, "r", encoding="utf-8") as f:
            data_list = json.load(f)

    if args.max_samples > 0:
        data_list = data_list[:args.max_samples]
        print(f"Using first {len(data_list)} samples (--max_samples={args.max_samples})")
    else:
        print(f"Loaded {len(data_list)} samples")
    return data_list


def compute_token_stats(tokenizer, model, prompt_text, generated_text):
    """Compute per-token log probabilities and entropy for the answer span only.

    Tokenizes prompt and full sequence separately to get exact prompt length, then
    runs a single forward pass and returns stats only for tokens after the prompt.

    Returns:
        tokens: list of token strings (shifted — each predicts the next token)
        target_ids: tensor of token IDs being predicted
        token_logprobs: log probability of each predicted token
        entropy: Shannon entropy of the full distribution at each position
        prompt_token_len: number of tokens belonging to the prompt
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_token_len = prompt_ids.shape[1]

    full_text = prompt_text + generated_text
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Shift: position t predicts token t+1
    target_ids = input_ids[:, 1:]
    log_probs = log_probs[:, :-1]
    probs = probs[:, :-1]

    token_logprobs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(target_ids.squeeze(0).tolist())

    return tokens, target_ids.squeeze(0), token_logprobs.squeeze(0), entropy.squeeze(0), prompt_token_len


def plot_logprob_distribution(all_logprobs, output_path):
    logprobs_array = np.array(all_logprobs)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(logprobs_array, bins=100, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Log Probability")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Log Probability Distribution", fontweight="bold")
    axes[0, 0].axvline(logprobs_array.mean(), color="red", linestyle="--", linewidth=2,
                       label=f"Mean: {logprobs_array.mean():.3f}")
    axes[0, 0].axvline(np.median(logprobs_array), color="green", linestyle="--", linewidth=2,
                       label=f"Median: {np.median(logprobs_array):.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    sorted_lp = np.sort(logprobs_array)
    cumulative = np.arange(1, len(sorted_lp) + 1) / len(sorted_lp)
    axes[0, 1].plot(sorted_lp, cumulative, linewidth=2, color="purple")
    axes[0, 1].set_xlabel("Log Probability")
    axes[0, 1].set_ylabel("Cumulative Probability")
    axes[0, 1].set_title("Cumulative Distribution Function", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].boxplot(logprobs_array, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="lightblue", alpha=0.7),
                       medianprops=dict(color="red", linewidth=2))
    axes[1, 0].set_ylabel("Log Probability")
    axes[1, 0].set_title("Box Plot", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].axis("off")
    stats_text = (
        f"    Statistics Summary\n"
        f"    {'═' * 27}\n\n"
        f"    Total Tokens: {len(logprobs_array):,}\n\n"
        f"    Mean:         {logprobs_array.mean():.4f}\n"
        f"    Median:       {np.median(logprobs_array):.4f}\n"
        f"    Std Dev:      {logprobs_array.std():.4f}\n\n"
        f"    Min:          {logprobs_array.min():.4f}\n"
        f"    Max:          {logprobs_array.max():.4f}\n\n"
        f"    Percentiles:\n"
        f"      25th:       {np.percentile(logprobs_array, 25):.4f}\n"
        f"      50th:       {np.percentile(logprobs_array, 50):.4f}\n"
        f"      75th:       {np.percentile(logprobs_array, 75):.4f}\n"
        f"      90th:       {np.percentile(logprobs_array, 90):.4f}\n"
        f"      95th:       {np.percentile(logprobs_array, 95):.4f}\n"
        f"      99th:       {np.percentile(logprobs_array, 99):.4f}\n"
    )
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family="monospace", verticalalignment="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved logprob distribution plot → {output_path}")
    plt.close()


def _plot_density_with_mean(ax, values, label, color, line_style="-", bins=120):
    if len(values) < 2:
        return
    arr = np.array(values, dtype=np.float32)
    hist, edges = np.histogram(arr, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(centers, hist, label=label, color=color, linestyle=line_style, linewidth=2)
    ax.axvline(arr.mean(), color=color, linestyle=":", linewidth=2, alpha=0.9)


def plot_special_token_distributions(all_logprobs, all_entropies, special_tokens, output_path):
    """Logprob and entropy density curves: all tokens vs. each tracked epistemic token."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _plot_density_with_mean(axes[0], all_logprobs, "All tokens", "#62c7a5", "-")
    _plot_density_with_mean(axes[1], all_entropies, "All tokens", "#62c7a5", "-")

    special_colors = ["#e53935", "#1e40ff", "#7e57c2", "#f59e0b", "#8d6e63",
                      "#00897b", "#f4511e", "#3949ab", "#6d4c41", "#039be5"]
    for idx, data in enumerate(special_tokens.values()):
        color = special_colors[idx % len(special_colors)]
        if data["logprobs"]:
            _plot_density_with_mean(axes[0], data["logprobs"], f"{data['token']}", color, "--")
        if data["entropies"]:
            _plot_density_with_mean(axes[1], data["entropies"], f"{data['token']}", color, "--")

    axes[0].set_title("Token Log Probability Density")
    axes[0].set_xlabel("Log Probability")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Token Entropy Density")
    axes[1].set_xlabel("Entropy")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved epistemic token distribution plot → {output_path}")
    plt.close()


def save_token_stats_csv(stats, output_path):
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "token_id", "count", "avg_logprob", "avg_entropy"])
        for token_id, s in sorted_stats:
            writer.writerow([
                s["token"], token_id, s["count"],
                s["logprob_sum"] / s["count"],
                s["entropy_sum"] / s["count"],
            ])
    print(f"Saved token statistics → {output_path}")


def save_special_tokens_csv(special_tokens, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "token_id", "instance_index", "logprob", "entropy"])
        for token_id, data in special_tokens.items():
            for idx, (lp, ent) in enumerate(zip(data["logprobs"], data["entropies"])):
                writer.writerow([data["token"], token_id, idx, lp, ent])
    print(f"Saved epistemic token occurrences → {output_path}")


def save_histogram_csv(values, output_path, bins=100, value_range=None, label="values"):
    arr = np.array(values)
    counts, bin_edges = np.histogram(arr, bins=bins, range=value_range)
    total_count = len(arr)
    bin_width = bin_edges[1] - bin_edges[0]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_start", "bin_end", "count", "density"])
        for i in range(len(counts)):
            density = counts[i] / (total_count * bin_width) if total_count > 0 else 0
            writer.writerow([f"{bin_edges[i]:.6f}", f"{bin_edges[i+1]:.6f}", counts[i], f"{density:.6f}"])
    print(f"Saved {label} histogram → {output_path}  ({total_count:,} tokens, {bins} bins)")


def print_special_token_summary(special_tokens):
    print("\n" + "=" * 60)
    print("Epistemic Token Statistics")
    print("=" * 60)
    found = [(tid, d) for tid, d in special_tokens.items() if d["logprobs"]]
    not_found = [(tid, d) for tid, d in special_tokens.items() if not d["logprobs"]]

    for token_id, data in sorted(found, key=lambda x: -len(x[1]["logprobs"])):
        print(f"\n  {data['token']:<16} (ID {token_id}):")
        print(f"    Count:        {len(data['logprobs'])}")
        print(f"    Mean logprob: {np.mean(data['logprobs']):.4f}  ±{np.std(data['logprobs']):.4f}")
        print(f"    Mean entropy: {np.mean(data['entropies']):.4f}  ±{np.std(data['entropies']):.4f}")

    if not_found:
        not_found_strs = ", ".join(d["token"] for _, d in not_found)
        print(f"\n  Not found in outputs: {not_found_strs}")


def main():
    args = parse_args()

    out_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    def out(filename):
        return os.path.join(out_dir, f"{args.tag}_{filename}")

    print(f"Loading model: {args.model_name_or_path}")
    tokenizer, model = load_model(args.model_name_or_path, args.adapter_path, args.dtype)

    print("Resolving epistemic token IDs...")
    special_tokens = resolve_special_token_ids(tokenizer, args.special_tokens)

    data_list = load_examples(args)

    stats = defaultdict(lambda: {"token": "", "count": 0, "logprob_sum": 0.0, "entropy_sum": 0.0})
    all_logprobs = []
    all_entropies = []

    for raw in tqdm(data_list, desc="Analyzing"):
        # Handle evaluate_aime.py results format (already has question/output flat)
        if "question" in raw and "output" in raw and len(raw) <= 3:
            data = {"system": "", "instruction": raw["question"], "input": "", "output": raw["output"]}
        else:
            data = normalize_example(raw, args)

        if not data["instruction"] or not data["output"]:
            continue

        prompt_text = build_chat_prompt(
            tokenizer,
            data.get("system", ""),
            data.get("instruction", ""),
            data.get("input", ""),
        )

        tokens, token_ids, logprobs, entropies, prompt_token_len = compute_token_stats(
            tokenizer, model, prompt_text, data.get("output", ""),
        )

        # After the shift, index (prompt_token_len - 1) is the first answer token.
        answer_start = prompt_token_len - 1

        for tok_id, lp, ent in zip(
            token_ids[answer_start:],
            logprobs[answer_start:],
            entropies[answer_start:],
        ):
            tok_id_int = int(tok_id)
            s = stats[tok_id_int]
            s["token"] = tokenizer.decode([tok_id_int])
            s["count"] += 1
            s["logprob_sum"] += float(lp)
            s["entropy_sum"] += float(ent)
            all_logprobs.append(float(lp))
            all_entropies.append(float(ent))

            if tok_id_int in special_tokens:
                special_tokens[tok_id_int]["logprobs"].append(float(lp))
                special_tokens[tok_id_int]["entropies"].append(float(ent))

    save_token_stats_csv(stats, out("token_stats.csv"))
    save_special_tokens_csv(special_tokens, out("special_tokens.csv"))
    save_histogram_csv(all_logprobs, out("all_logprobs.csv"), bins=100, value_range=(-30, 0), label="logprob")
    save_histogram_csv(all_entropies, out("all_entropies.csv"), bins=100, value_range=(0, 10), label="entropy")
    print_special_token_summary(special_tokens)

    print("\nPlotting overall logprob distribution...")
    plot_logprob_distribution(all_logprobs, out("logprob_dist.png"))
    print("Plotting epistemic token logprob/entropy distributions...")
    plot_special_token_distributions(all_logprobs, all_entropies, special_tokens, out("special_token_dist.png"))

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
