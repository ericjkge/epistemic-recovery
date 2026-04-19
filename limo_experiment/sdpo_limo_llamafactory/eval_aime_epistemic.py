#!/usr/bin/env python3
"""
Evaluate SDPO Qwen3-8B LoRA adapter on AIME accuracy + epistemic verbalization.

Epistemic tracking mirrors limo_lora_sft.py. AIME loading mirrors strategic
codebase eval/eval.py. Compares adapter vs base model when --compare_base is set.
"""
import argparse, json, os, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

EPISTEMIC_TOKENS = [
    "wait", "hmm", "perhaps", "maybe", "actually", "alternatively",
    "seems", "might", "likely", "check", "let me", "reconsider",
    "hold on", "no,", "but wait",
]

SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. Think through the problem carefully, "
    "then provide the final answer within \\boxed{}."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aime_problems(data_path: str = None) -> list[dict]:
    """Load AIME problems from: user-provided path → strategic codebase → HuggingFace."""
    # 1. User-provided path
    if data_path and os.path.exists(data_path):
        with open(data_path) as f:
            return [json.loads(l) for l in f if l.strip()] if data_path.endswith(".jsonl") else json.load(f)

    # 2. Strategic codebase eval/data/aime/
    strategic_aime = (
        Path(__file__).parent.parent
        / "strategic-information-allocation-llm-reasoning"
        / "eval"
        / "data"
        / "aime"
    )
    if strategic_aime.exists():
        files = sorted(strategic_aime.glob("*.json")) + sorted(strategic_aime.glob("*.jsonl"))
        if files:
            problems = []
            for f in files:
                with open(f) as fh:
                    data = [json.loads(l) for l in fh if l.strip()] if str(f).endswith(".jsonl") else json.load(fh)
                problems.extend(data if isinstance(data, list) else [data])
            print(f"Loaded {len(problems)} AIME problems from {strategic_aime}")
            return problems

    # 3. HuggingFace fallback (AI-MO/aimo-validation-aime)
    try:
        from datasets import load_dataset
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        problems = [{"problem": r["problem"], "answer": str(r["answer"])} for r in ds]
        print(f"Loaded {len(problems)} AIME problems from HuggingFace (AI-MO/aimo-validation-aime)")
        return problems
    except Exception as e:
        print(f"HuggingFace fallback failed: {e}")

    raise RuntimeError(
        "No AIME data found. Pass --aime_data <path> or place data in "
        "strategic-information-allocation-llm-reasoning/eval/data/aime/"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def count_epistemic(text: str) -> dict:
    text_lower = text.lower()
    words = text.split()
    total = sum(text_lower.count(t) for t in EPISTEMIC_TOKENS)
    return {
        "count": total,
        "per_1k_tokens": total / max(len(words), 1) * 1000,
        "length": len(words),
    }


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else ""


def normalize(answer: str) -> str:
    try:
        return str(int(float(answer.strip())))
    except (ValueError, OverflowError):
        return answer.strip()


# ---------------------------------------------------------------------------
# Model loading / generation
# ---------------------------------------------------------------------------

def load_model(base_model: str, adapter_path: str | None, load_in_4bit: bool):
    print(f"Loading {base_model}" + (f" + adapter {adapter_path}" if adapter_path else " (base only)"))
    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_cfg,
        attn_implementation="flash_attention_2" if not load_in_4bit else None,
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def generate_response(
    tokenizer, model, question: str, max_new_tokens: int, temperature: float
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": question + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(tokenizer, model, problems: list[dict], max_new_tokens: int, temperature: float):
    results = []
    n_correct = 0
    n_with_gt = 0

    for i, prob in enumerate(problems):
        q_key = next((k for k in ["question", "problem", "input"] if k in prob), None)
        a_key = next((k for k in ["answer", "final_answer", "label"] if k in prob), None)
        if q_key is None:
            continue

        question = prob[q_key]
        ground_truth = str(prob[a_key]).strip() if a_key else None

        print(f"\n[{i+1}/{len(problems)}] Generating...", flush=True)
        response = generate_response(tokenizer, model, question, max_new_tokens, temperature)

        pred = extract_boxed(response)
        is_correct = None
        if ground_truth is not None:
            is_correct = normalize(pred) == normalize(ground_truth)
            n_with_gt += 1
            if is_correct:
                n_correct += 1

        ep = count_epistemic(response)
        results.append(
            {
                "problem": question[:120] + "...",
                "ground_truth": ground_truth,
                "predicted": pred,
                "correct": is_correct,
                "response_length": ep["length"],
                "epistemic_count": ep["count"],
                "epistemic_per_1k": ep["per_1k_tokens"],
                "response_preview": response[:300],
            }
        )

        status = ("✓" if is_correct else "✗") if is_correct is not None else "?"
        print(
            f"  {status} pred={pred!r:>8}  gt={str(ground_truth)!r:<8}  "
            f"ep/1k={ep['per_1k_tokens']:.2f}  len={ep['length']}"
        )

    accuracy = n_correct / n_with_gt if n_with_gt else None
    avg_ep = sum(r["epistemic_per_1k"] for r in results) / len(results) if results else 0
    avg_len = sum(r["response_length"] for r in results) / len(results) if results else 0

    summary = {
        "n_problems": len(results),
        "n_correct": n_correct,
        "n_with_ground_truth": n_with_gt,
        "accuracy": accuracy,
        "avg_epistemic_per_1k": avg_ep,
        "avg_response_length": avg_len,
    }
    return results, summary


def print_summary(label: str, summary: dict):
    acc = f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "N/A"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Accuracy:             {acc} ({summary['n_correct']}/{summary['n_with_ground_truth']})")
    print(f"  Avg epistemic/1k:     {summary['avg_epistemic_per_1k']:.3f}")
    print(f"  Avg response length:  {summary['avg_response_length']:.0f} tokens")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AIME accuracy + epistemic verbalization eval")
    parser.add_argument(
        "--base_model", default="beanie00/math-SDPO-Qwen3-8B-think-off-step-100"
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Path to LlamaFactory LoRA output dir (e.g. outputs/sdpo-qwen3-8b-limo-lora-r16)",
    )
    parser.add_argument("--aime_data", default=None, help="Path to AIME JSON/JSONL file")
    parser.add_argument("--n_problems", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="Also evaluate base model without adapter for delta comparison",
    )
    parser.add_argument("--output_json", default="eval_results.json")
    args = parser.parse_args()

    problems = load_aime_problems(args.aime_data)
    if args.n_problems:
        problems = problems[: args.n_problems]
    print(f"Evaluating on {len(problems)} problems | max_new_tokens={args.max_new_tokens} | temp={args.temperature}")

    all_results = {}

    # --- Adapter evaluation ---
    adapter_label = "adapter" if args.adapter_path else "base_no_adapter"
    tok, mdl = load_model(args.base_model, args.adapter_path, args.load_in_4bit)
    results, summary = run_eval(tok, mdl, problems, args.max_new_tokens, args.temperature)
    all_results[adapter_label] = {"summary": summary, "results": results}
    print_summary(adapter_label, summary)

    # --- Base model comparison ---
    if args.compare_base and args.adapter_path:
        del tok, mdl
        torch.cuda.empty_cache()
        print("\nLoading base model for comparison...")
        tok_b, mdl_b = load_model(args.base_model, None, args.load_in_4bit)
        results_b, summary_b = run_eval(tok_b, mdl_b, problems, args.max_new_tokens, args.temperature)
        all_results["base"] = {"summary": summary_b, "results": results_b}
        print_summary("base", summary_b)

        s, sb = summary, summary_b
        print(f"\n{'='*60}")
        print("  DELTA (adapter − base)")
        if s["accuracy"] is not None and sb["accuracy"] is not None:
            print(f"  Accuracy:            {s['accuracy']-sb['accuracy']:+.1%}")
        print(f"  Epistemic/1k:        {s['avg_epistemic_per_1k']-sb['avg_epistemic_per_1k']:+.3f}")
        print(f"  Response length:     {s['avg_response_length']-sb['avg_response_length']:+.0f} tokens")

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {args.output_json}")


if __name__ == "__main__":
    main()
