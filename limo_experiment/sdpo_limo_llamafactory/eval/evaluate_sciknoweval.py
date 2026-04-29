#!/usr/bin/env python3
"""Evaluate Qwen3-8B variants on SciKnowEval chemistry MCQs.

Companion to evaluate_aime.py.  The point of this eval is to check that LoRA
fine-tuning on LIMO math reasoning preserves chemistry knowledge — i.e. that
gains on out-of-domain math (AIME) don't come at the cost of in-domain
chemistry MCQ accuracy.

Data: data/sciknoweval_chemistry.jsonl  (one JSON object per line; 210 4-way MCQs)
Each record carries a `system` prompt that asks for a <reasoning>...</reasoning>
<answer>X</answer> response, a `prompt` containing the question + options A–D,
and an `answer` letter.

Sampling, loop detection, vLLM/HF backend selection, and adapter handling are
imported from evaluate_aime so behaviour stays identical across benchmarks.

Run from sdpo_limo_llamafactory/:
    python eval/evaluate_sciknoweval.py --adapter_path saves/qwen3_sdpo_limo_lora
"""
import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch
except Exception:
    torch = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except Exception:
    LLM = None
    SamplingParams = None
    LoRARequest = None

from evaluate_aime import BASE_MODEL, PRETRAINED


_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_DATA = _DATA_DIR / "sciknoweval_chemistry.jsonl"

# Primary extraction target — the system prompt asks for this exact tag form.
_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*([A-Da-d])\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
# Fallbacks for when the model didn't follow format (still common with reasoning models).
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Da-d])\s*\}")
_FINAL_LETTER_RE = re.compile(
    r"(?:final answer|the answer is|answer\s*[:=])\s*\(?([A-Da-d])\)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sciknoweval_chemistry(path: Path):
    """Load the JSONL chemistry MCQ file. Returns a list of normalized dicts."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Sanity check the fields we depend on.
            if "prompt" not in r or "answer" not in r:
                continue
            ans = str(r["answer"]).strip().upper()
            if ans not in {"A", "B", "C", "D"}:
                continue
            records.append({
                "problem_id": f"chem_{r.get('idx', i):05d}",
                "system": r.get("system", "").strip(),
                "question": r["prompt"],
                "answer": ans,
            })
    return records


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def extract_letter(text: str) -> str:
    """Return 'A'/'B'/'C'/'D' or '' if no letter could be confidently extracted."""
    if not text:
        return ""
    # 1. Tagged answer (system-prompt format).
    m = None
    for m_ in _ANSWER_TAG_RE.finditer(text):
        m = m_
    if m:
        return m.group(1).upper()
    # 2. \boxed{X}
    for m_ in _BOXED_LETTER_RE.finditer(text):
        m = m_
    if m:
        return m.group(1).upper()
    # 3. "Final answer: X" / "The answer is X"
    for m_ in _FINAL_LETTER_RE.finditer(text):
        m = m_
    if m:
        return m.group(1).upper()
    return ""


def grade_letter(extracted: str, ground_truth: str) -> bool:
    return bool(extracted) and extracted.upper() == ground_truth.upper()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompts(tokenizer, problems, enable_thinking: bool):
    prompts = []
    for p in problems:
        messages = []
        if p["system"]:
            messages.append({"role": "system", "content": p["system"]})
        messages.append({"role": "user", "content": p["question"]})
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            prompts.append(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompts.append(tokenizer.apply_chat_template(messages, **kwargs))
    return prompts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one_model(
    label: str,
    model_path: str,
    adapter_path: str | None,
    problems: list,
    sampling_params: Any,
    results_dir: Path,
    tensor_parallel_size: int,
    max_model_len: int,
    enable_thinking: bool,
    gpu_memory_utilization: float,
    backend: str,
    benchmark_label: str = "sciknoweval_chemistry",
):
    print(f"\n{'='*70}\nLoading {label}: {model_path}" + (f" + LoRA {adapter_path}" if adapter_path else ""))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if backend == "vllm":
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vLLM backend requested but vllm is not installed.")
        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            enforce_eager=True,
        )
        lora_request = None
        if adapter_path:
            llm_kwargs.update(enable_lora=True, max_lora_rank=16, max_loras=1)
            lora_request = LoRARequest("limo_adapter", 1, adapter_path)
        model_runner = LLM(**llm_kwargs)
    else:
        if torch is None:
            raise RuntimeError("HF backend requested but torch is not installed.")
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model_runner = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch_dtype, device_map="auto",
            )
        except ValueError as e:
            if "requires `accelerate`" not in str(e):
                raise
            model_runner = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch_dtype,
            ).to(hf_device)
        if adapter_path:
            if PeftModel is None:
                raise RuntimeError("HF LoRA path requested but peft is not installed.")
            model_runner = PeftModel.from_pretrained(model_runner, adapter_path)
            if not hasattr(model_runner, "hf_device_map"):
                model_runner = model_runner.to(hf_device)
        model_runner.eval()

    print(f"\n[{label} / {benchmark_label}] generating n={sampling_params.n} for {len(problems)} problems...")
    prompts = build_prompts(tokenizer, problems, enable_thinking=enable_thinking)

    if backend == "vllm":
        gen_kwargs = {}
        if lora_request is not None:
            gen_kwargs["lora_request"] = lora_request
        completions = model_runner.generate(prompts, sampling_params, **gen_kwargs)
    else:
        completions = []
        if hasattr(model_runner, "hf_device_map"):
            hf_input_device = next(model_runner.parameters()).device
        else:
            hf_input_device = hf_device
        for prompt in tqdm(prompts, desc="  generating", unit="problem"):
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(hf_input_device) for k, v in encoded.items()}
            input_len = encoded["input_ids"].shape[1]
            hf_gen_kwargs = dict(
                do_sample=True,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                max_new_tokens=sampling_params.max_tokens,
                num_return_sequences=sampling_params.n,
                pad_token_id=tokenizer.eos_token_id,
            )
            with torch.no_grad():
                generated = model_runner.generate(**encoded, **hf_gen_kwargs)
            outputs = []
            for seq in generated:
                seq = seq[input_len:]
                outputs.append({
                    "text": tokenizer.decode(seq, skip_special_tokens=True),
                    "token_ids": seq.tolist(),
                })
            completions.append({"outputs": outputs})

    results = []
    n_correct_problems = 0
    for prob, comp in tqdm(zip(problems, completions), desc="  grading", total=len(problems), unit="problem"):
        if backend == "vllm":
            generations = [o.text for o in comp.outputs]
            response_lengths = [len(o.token_ids) for o in comp.outputs]
        else:
            generations = [o["text"] for o in comp["outputs"]]
            response_lengths = [len(o["token_ids"]) for o in comp["outputs"]]
        extracted = [extract_letter(g) for g in generations]
        correctness = [grade_letter(e, prob["answer"]) for e in extracted]
        if any(correctness):
            n_correct_problems += 1
        results.append({
            "problem_id": prob["problem_id"],
            "question": prob["question"],
            "ground_truth": prob["answer"],
            "generations": generations,
            "extracted_answers": extracted,
            "correctness": correctness,
            "response_lengths": response_lengths,
        })

    flat = [c for r in results for c in r["correctness"]]
    acc_at_n = sum(flat) / len(flat) if flat else 0.0
    any_correct_rate = n_correct_problems / len(results) if results else 0.0
    n_unparsed = sum(1 for r in results for e in r["extracted_answers"] if not e)
    print(
        f"  acc@{sampling_params.n} (mean): {acc_at_n:.3f}    "
        f"any-correct rate: {any_correct_rate:.3f}    "
        f"({n_correct_problems}/{len(results)})    "
        f"unparsed: {n_unparsed}/{len(results)*sampling_params.n}"
    )

    out_path = results_dir / f"{label}_{benchmark_label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "label": label,
            "model": model_path,
            "adapter": adapter_path,
            "benchmark": benchmark_label,
            "n_sampling": sampling_params.n,
            "acc_mean": acc_at_n,
            "any_correct_rate": any_correct_rate,
            "n_unparsed": n_unparsed,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  wrote {out_path}")

    del model_runner
    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="saves/qwen3_sdpo_limo_lora")
    parser.add_argument("--lora_merged_path", default="saves/qwen3_sdpo_limo_lora_merged",
                        help="If this path exists, the 'lora' model loads it as a merged checkpoint "
                             "instead of base+adapter (vLLM Punica workaround).")
    parser.add_argument("--data_path", default=str(_DEFAULT_DATA),
                        help="JSONL of MCQs (one record per line).")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_sampling", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max new tokens. MCQ chemistry rarely needs >4K of legitimate "
                             "reasoning; 8K leaves headroom and matches the loop-detector calibration.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--tensor_parallel_size", type=int,
                        default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--models", default="baseline,lora",
                        help="Comma-separated subset of {baseline,lora,pretrained}. "
                             "Pretrained is off by default since the OOD-vs-in-domain story "
                             "only needs baseline-vs-LoRA.")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--max_questions", type=int, default=0,
                        help="If > 0, evaluate only the first N questions.")

    args = parser.parse_args()

    print(f"Loading SciKnowEval chemistry from {args.data_path}...")
    problems = load_sciknoweval_chemistry(Path(args.data_path))
    if args.max_questions and args.max_questions > 0:
        problems = problems[:args.max_questions]
    print(f"  {len(problems)} problems")

    if args.backend == "vllm":
        if SamplingParams is None:
            raise RuntimeError("vLLM backend requested but vllm is not installed.")
        sampling_params = SamplingParams(
            n=args.n_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            frequency_penalty=args.frequency_penalty,
        )
    else:
        sampling_params = SimpleNamespace(
            n=args.n_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            frequency_penalty=args.frequency_penalty,
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    requested = {m.strip() for m in args.models.split(",") if m.strip()}

    if "baseline" in requested:
        run_one_model(
            label="baseline_sdpo_think_step100",
            model_path=BASE_MODEL,
            adapter_path=None,
            problems=problems,
            sampling_params=sampling_params,
            results_dir=results_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_thinking=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            backend=args.backend,
        )

    if "lora" in requested:
        lora_merged = Path(args.lora_merged_path)
        if lora_merged.exists() and (lora_merged / "config.json").exists():
            lora_model_path = str(lora_merged)
            lora_adapter_path = None
            print(f"  loading lora as merged checkpoint: {lora_model_path}")
        else:
            lora_model_path = BASE_MODEL
            lora_adapter_path = args.adapter_path
        run_one_model(
            label="lora_sdpo_plus_limo",
            model_path=lora_model_path,
            adapter_path=lora_adapter_path,
            problems=problems,
            sampling_params=sampling_params,
            results_dir=results_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_thinking=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            backend=args.backend,
        )

    if "pretrained" in requested:
        run_one_model(
            label="pretrained_qwen3_8b",
            model_path=PRETRAINED,
            adapter_path=None,
            problems=problems,
            sampling_params=sampling_params,
            results_dir=results_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_thinking=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()
