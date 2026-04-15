"""
AIME Epistemic Evaluation Script
=================================
Runs vLLM inference on AIME 2024 / AIME 2025, computes:
  - Pass@1 accuracy
  - Per-response and per-problem epistemic token counts

Usage:
  # Pre-LoRA (base model):
  python aime_epistemic_eval.py \
      --model_path beanie00/math-SDPO-Qwen3-8B-think-step-100 \
      --aime_years 2024 2025 \
      --output_dir ./results/pre_lora \
      --enable_thinking

  # Post-LoRA (merged model):
  python aime_epistemic_eval.py \
      --model_path ./lora_outputs/merged \
      --aime_years 2024 2025 \
      --output_dir ./results/post_lora \
      --enable_thinking

Reuses grader / parser utilities from self-distillation-analysis/eval/utils/.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# ── Path to the existing repo utilities ─────────────────────────────────────
REPO_EVAL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "eval",
)
sys.path.insert(0, os.path.abspath(REPO_EVAL_DIR))

from utils.grader import check_is_correct
from utils.parser import extract_answer

# ── Epistemic token list (from check_epistemic_tokens.py) ───────────────────
EPISTEMIC_WORDS = [
    "wait", "hmm", "perhaps", "maybe", "actually",
    "alternatively", "seems", "might", "likely", "check",
]

AIME_HF_DATASETS = {
    2024: "math-ai/aime24",
    2025: "math-ai/aime25",
}

# Map HF dataset → column names
AIME_COLUMN_MAP = {
    "math-ai/aime24":  {"problem": "problem", "answer": "solution"},   # answer is in 'solution', wrapped \boxed{}
    "math-ai/aime25":  {"problem": "problem", "answer": "answer"},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def count_epistemic_words(text: str) -> tuple[dict, int]:
    text_lower = text.lower()
    counts = {w: text_lower.count(w) for w in EPISTEMIC_WORDS}
    return counts, sum(counts.values())


def load_aime_from_parquet(parquet_path: str) -> list[dict]:
    """Load from the repo's pre-processed parquet (data/math/evaluation/)."""
    df = pd.read_parquet(parquet_path)
    return df.to_dict(orient="records")


def load_aime_from_hf(year: int) -> list[dict]:
    """Pull AIME directly from HuggingFace Datasets."""
    from datasets import load_dataset

    ds_name = AIME_HF_DATASETS[year]
    col_map = AIME_COLUMN_MAP[ds_name]
    ds = load_dataset(ds_name, split="test")

    examples = []
    for ex in ds:
        problem = ex[col_map["problem"]]
        raw_answer = str(ex[col_map["answer"]])
        # AIME 2024 wraps the answer in \boxed{...}
        if ds_name == "math-ai/aime24":
            m = re.search(r"\\boxed\{(.+?)\}", raw_answer)
            answer = m.group(1) if m else raw_answer
        else:
            answer = raw_answer
        examples.append({"problem": problem, "answer": answer, "source": ds_name})
    return examples


def build_prompt(tokenizer, problem: str, enable_thinking: bool, thinking_budget=None) -> str:
    """Apply the model's chat template with a standard math system prompt."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math reasoning assistant. "
                       "Solve the problem step by step and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": problem,
        },
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        kwargs["enable_thinking"] = True
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
    else:
        try:
            kwargs["enable_thinking"] = False
        except Exception:
            pass

    return tokenizer.apply_chat_template(messages, **kwargs)


def run_vllm_inference(
    model_path: str,
    prompts: list[str],
    n_samples: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
) -> list:
    from vllm import LLM, SamplingParams

    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    tp_size = len(available_gpus)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
    )
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k if temperature > 0 else -1,
    )
    return llm.generate(prompts, sampling_params)


# ── Main evaluation function ─────────────────────────────────────────────────

def evaluate(args):
    from transformers import AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    tag = "think" if args.enable_thinking else "nothink"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    all_examples: list[dict] = []
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    for year in args.aime_years:
        # Try local parquet first (already in the repo)
        # Files are named aime24.parquet, aime25.parquet (2-digit year suffix)
        short_year = str(year)[-2:]
        local_parquet = os.path.join(repo_root, "data", "math", "evaluation", f"aime{short_year}.parquet")
        if os.path.exists(local_parquet):
            print(f"Loading AIME {year} from local parquet: {local_parquet}")
            rows = load_aime_from_parquet(local_parquet)
            # Normalise column names from the veRL parquet format
            for row in rows:
                # veRL parquet format: prompt (list of msgs), reward_model (dict with ground_truth)
                if "prompt" in row and "reward_model" in row:
                    msgs = row["prompt"]
                    if isinstance(msgs, str):
                        msgs = json.loads(msgs)
                    problem_text = next(
                        (m["content"] for m in msgs if m["role"] == "user"), msgs[-1]["content"]
                    )
                    rm = row["reward_model"]
                    if isinstance(rm, str):
                        rm = json.loads(rm)
                    answer = str(rm["ground_truth"])
                    all_examples.append({"problem": problem_text, "answer": answer, "source": f"aime{year}"})
                else:
                    all_examples.append(row)
        else:
            print(f"Local parquet not found; downloading AIME {year} from HuggingFace …")
            all_examples.extend(load_aime_from_hf(year))

    if args.end_idx != -1:
        all_examples = all_examples[args.start_idx : args.end_idx]
    else:
        all_examples = all_examples[args.start_idx :]
    print(f"Total problems: {len(all_examples)}")

    # ── 2. Build prompts ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [
        build_prompt(tokenizer, ex["problem"], args.enable_thinking, args.thinking_budget)
        for ex in tqdm(all_examples, desc="Building prompts")
    ]
    print(f"\nSample prompt (first 500 chars):\n{prompts[0][:500]}\n{'='*60}")

    # ── 3. vLLM inference ─────────────────────────────────────────────────────
    temperature = args.temperature
    top_p = 1.0 if temperature == 0 else args.top_p
    top_k = -1 if temperature == 0 else args.top_k

    completions = run_vllm_inference(
        model_path=args.model_path,
        prompts=prompts,
        n_samples=args.n_samples,
        temperature=temperature,
        max_tokens=args.max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    # ── 4. Parse, grade, epistemic counts ────────────────────────────────────
    results = []
    correct_total = 0

    # Aggregate epistemic stats
    global_word_counts: dict[str, int] = defaultdict(int)
    global_epistemic_total = 0
    total_responses = 0
    total_tokens = 0

    for i, ex in enumerate(tqdm(all_examples, desc="Grading")):
        gt_answer = str(ex["answer"])
        raw_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]

        # Separate thinking vs. answer parts
        thinking_parts: list[str] = []
        answer_parts: list[str] = []
        if args.enable_thinking:
            for resp in raw_responses:
                m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
                if m:
                    thinking_parts.append(m.group(1).strip())
                    answer_parts.append(resp[m.end():].strip())
                else:
                    thinking_parts.append("")
                    answer_parts.append(resp)
        else:
            answer_parts = raw_responses
            thinking_parts = [""] * len(raw_responses)

        # Token lengths
        response_token_lengths = [
            len(tokenizer.encode(r, add_special_tokens=False)) for r in raw_responses
        ]
        thinking_token_lengths = [
            len(tokenizer.encode(t, add_special_tokens=False)) if t else 0
            for t in thinking_parts
        ]
        total_tokens += sum(response_token_lengths)
        total_responses += len(raw_responses)

        # Epistemic counts (on full response including thinking)
        prob_word_counts: dict[str, int] = defaultdict(int)
        prob_epistemic_total = 0
        for resp in raw_responses:
            wc, wt = count_epistemic_words(resp)
            for w, c in wc.items():
                prob_word_counts[w] += c
                global_word_counts[w] += c
            prob_epistemic_total += wt
            global_epistemic_total += wt

        # Grading
        extracted_answers = [extract_answer(ap, "math") for ap in answer_parts]
        correctness = [check_is_correct(ea, gt_answer) for ea in extracted_answers]
        is_correct = any(correctness)
        if is_correct:
            correct_total += 1

        results.append({
            "problem": ex["problem"],
            "gold_answer": gt_answer,
            "source": ex.get("source", ""),
            "generated_responses": raw_responses,
            "answer_parts": answer_parts,
            "thinking_parts": thinking_parts,
            "extracted_answers": extracted_answers,
            "correctness": correctness,
            "is_correct": is_correct,
            "response_token_lengths": response_token_lengths,
            "thinking_token_lengths": thinking_token_lengths,
            "avg_response_tokens": sum(response_token_lengths) / len(response_token_lengths),
            "avg_thinking_tokens": (
                sum(thinking_token_lengths) / len(thinking_token_lengths) if thinking_token_lengths else 0
            ),
            "epistemic_word_counts": dict(prob_word_counts),
            "epistemic_total": prob_epistemic_total,
            "avg_epistemic_per_response": prob_epistemic_total / len(raw_responses),
        })

    # ── 5. Summary ────────────────────────────────────────────────────────────
    n = len(all_examples)
    accuracy = correct_total / n if n else 0
    avg_resp_tokens = total_tokens / total_responses if total_responses else 0
    avg_epistemic_per_resp = global_epistemic_total / total_responses if total_responses else 0
    avg_epistemic_per_prob = global_epistemic_total / n if n else 0

    print(f"\n{'='*60}")
    print(f"Model : {args.model_path}")
    print(f"AIME years: {args.aime_years}  |  Problems: {n}  |  Responses/problem: {args.n_samples}")
    print(f"Thinking mode: {'ON' if args.enable_thinking else 'OFF'}")
    print(f"\nAccuracy (Pass@1): {accuracy:.4f}  ({correct_total}/{n})")
    print(f"Avg response tokens: {avg_resp_tokens:.1f}")
    if args.enable_thinking:
        all_think_lengths = [t for r in results for t in r["thinking_token_lengths"]]
        if all_think_lengths:
            print(f"Avg thinking tokens: {sum(all_think_lengths)/len(all_think_lengths):.1f}")
    print(f"\nEpistemic Token Counts (over ALL responses):")
    print(f"{'Word':<18} {'Total':>8} {'Avg/Resp':>10} {'Avg/Prob':>10}")
    print("─" * 50)
    for w in EPISTEMIC_WORDS:
        t = global_word_counts[w]
        print(f"{w:<18} {t:>8} {t/total_responses if total_responses else 0:>10.3f} {t/n if n else 0:>10.3f}")
    print("─" * 50)
    print(
        f"{'TOTAL':<18} {global_epistemic_total:>8} "
        f"{avg_epistemic_per_resp:>10.2f} {avg_epistemic_per_prob:>10.2f}"
    )
    print(f"{'='*60}\n")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    model_tag = args.model_path.rstrip("/").split("/")[-1]
    years_tag = "_".join(str(y) for y in args.aime_years)
    out_jsonl = os.path.join(
        args.output_dir,
        f"aime{years_tag}_{model_tag}_{tag}_t{temperature}_n{args.n_samples}_{timestamp}.jsonl",
    )
    out_summary = out_jsonl.replace(".jsonl", "_summary.json")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results saved → {out_jsonl}")

    summary = {
        "model": args.model_path,
        "aime_years": args.aime_years,
        "n_problems": n,
        "n_samples": args.n_samples,
        "enable_thinking": args.enable_thinking,
        "temperature": temperature,
        "accuracy_pass_at_1": accuracy,
        "correct_count": correct_total,
        "avg_response_tokens": avg_resp_tokens,
        "epistemic": {
            "total": global_epistemic_total,
            "avg_per_response": avg_epistemic_per_resp,
            "avg_per_problem": avg_epistemic_per_prob,
            "per_word": {w: global_word_counts[w] for w in EPISTEMIC_WORDS},
        },
        "timestamp": timestamp,
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved  → {out_summary}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AIME epistemic evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--aime_years", type=int, nargs="+", default=[2024, 2025],
                        help="Which AIME years to evaluate (2024 and/or 2025)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples per problem (pass@n)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--enable_thinking", action="store_true", default=True,
                        help="Enable <think> mode for Qwen3 / DeepSeek-R1 style models")
    parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--thinking_budget", type=int, default=None,
                        help="Optional max thinking tokens (model-dependent)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
