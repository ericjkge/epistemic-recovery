#!/usr/bin/env python3
"""Evaluate three Qwen3-8B variants on AIME24 + AIME25 with vLLM.

Models:
  baseline          beanie00/math-SDPO-Qwen3-8B-think-step-100  (no adapter)
  lora              same base + LoRA adapter from saves/qwen3_sdpo_limo_lora/
  pretrained        Qwen/Qwen3-8B  (upper bound, full thinking-mode pretrain)

For each (model, benchmark) we sample n=16 completions per problem with the Kim et al.
generation settings, extract the LAST \\boxed{...} as the answer, and grade against the
integer ground truth. Per-problem results are saved to results/{model}_{benchmark}.json.
"""
import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


BASE_MODEL = "beanie00/math-SDPO-Qwen3-8B-think-step-100"
PRETRAINED = "Qwen/Qwen3-8B"

USER_TEMPLATE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_aime_record(rec):
    """Pull (question, answer) out of various AIME schema variants."""
    q = (rec.get("problem") or rec.get("question") or rec.get("Problem") or
         rec.get("Question") or rec.get("query"))
    a = (rec.get("answer") or rec.get("Answer") or rec.get("solution_answer") or
         rec.get("final_answer"))
    return q, a


def load_aime_benchmark(name: str):
    """Return a list of {id, question, answer:int} dicts.

    name in {"aime24", "aime25"}. Tries the spec's preferred HF datasets first and
    falls back to common alternates, reporting which path succeeded.
    """
    from datasets import load_dataset

    candidates = {
        "aime24": [
            ("HuggingFaceH4/aime_2024", "train"),
            ("Maxwell-Jia/AIME_2024", "train"),
            ("AI-MO/aimo-validation-aime", "train"),
        ],
        "aime25": [
            ("opencompass/AIME2025", "test"),
            ("yentinglin/aime_2025", "train"),
            ("MathArena/aime_2025", "train"),
        ],
    }[name]

    last_err = None
    for ds_name, split in candidates:
        try:
            ds = load_dataset(ds_name, split=split)
            print(f"  loaded {name} from {ds_name} ({split}) — {len(ds)} problems")
            out = []
            for i, row in enumerate(ds):
                q, a = _normalize_aime_record(row)
                if q is None or a is None:
                    continue
                try:
                    a_int = int(str(a).strip())
                except ValueError:
                    continue
                out.append({"problem_id": f"{name}_{i:03d}", "question": q, "answer": a_int})
            return out
        except Exception as e:
            last_err = e
            print(f"  {ds_name} failed: {e}")
    raise RuntimeError(f"Could not load {name} from any candidate. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def extract_last_boxed(text: str) -> str:
    """Return the inner contents of the LAST \\boxed{...} in text, or ''. """
    last = None
    for m in BOXED_RE.finditer(text):
        last = m
    return last.group(1).strip() if last else ""


def grade_int(extracted: str, ground_truth: int) -> bool:
    if not extracted:
        return False
    # Strip latex wrappers like \text{42}, $42$, leading "+".
    s = extracted.replace("\\,", "").replace(",", "").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.strip("$ \t").lstrip("+")
    # Pull the first integer-looking substring (handles "42." or "42 " etc.).
    m = re.search(r"-?\d+", s)
    if not m:
        return False
    try:
        return int(m.group()) == ground_truth
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# vLLM driver
# ---------------------------------------------------------------------------

def build_prompts(tokenizer, problems, enable_thinking: bool):
    prompts = []
    for p in problems:
        messages = [
            {"role": "user", "content": USER_TEMPLATE.format(question=p["question"])},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            prompts.append(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompts.append(tokenizer.apply_chat_template(messages, **kwargs))
    return prompts


def run_one_model(
    label: str,
    model_path: str,
    adapter_path: str | None,
    benchmarks: dict,
    sampling_params: Any,
    results_dir: Path,
    tensor_parallel_size: int,
    max_model_len: int,
    enable_thinking: bool,
    gpu_memory_utilization: float,
    backend: str,
):
    print(f"\n{'='*70}\nLoading {label}: {model_path}" + (f" + LoRA {adapter_path}" if adapter_path else ""))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if backend == "vllm":
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vLLM backend requested but vllm is not installed in this environment.")

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
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
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        except ValueError as e:
            if "requires `accelerate`" not in str(e):
                raise
            print("  [hf] accelerate not found; falling back to single-device loading.")
            model_runner = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            model_runner = model_runner.to(hf_device)
        if adapter_path:
            if PeftModel is None:
                raise RuntimeError("HF LoRA path requested but peft is not installed.")
            model_runner = PeftModel.from_pretrained(model_runner, adapter_path)
            if not hasattr(model_runner, "hf_device_map"):
                model_runner = model_runner.to(hf_device)
        model_runner.eval()

    for bench_name, problems in benchmarks.items():
        print(f"\n[{label} / {bench_name}] generating n={sampling_params.n} for {len(problems)} problems...")
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
            for prompt in prompts:
                encoded = tokenizer(prompt, return_tensors="pt")
                encoded = {k: v.to(hf_input_device) for k, v in encoded.items()}
                input_len = encoded["input_ids"].shape[1]

                with torch.no_grad():
                    generated = model_runner.generate(
                        **encoded,
                        do_sample=True,
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                        top_k=sampling_params.top_k,
                        max_new_tokens=sampling_params.max_tokens,
                        num_return_sequences=sampling_params.n,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                outputs = []
                for seq in generated:
                    seq = seq[input_len:]
                    outputs.append(
                        {
                            "text": tokenizer.decode(seq, skip_special_tokens=True),
                            "token_ids": seq.tolist(),
                        }
                    )
                completions.append({"outputs": outputs})

        results = []
        n_correct_problems = 0
        for prob, comp in zip(problems, completions):
            if backend == "vllm":
                generations = [o.text for o in comp.outputs]
                response_lengths = [len(o.token_ids) for o in comp.outputs]
            else:
                generations = [o["text"] for o in comp["outputs"]]
                response_lengths = [len(o["token_ids"]) for o in comp["outputs"]]
            extracted = [extract_last_boxed(g) for g in generations]
            correctness = [grade_int(e, prob["answer"]) for e in extracted]
            if any(correctness):
                n_correct_problems += 1
            results.append(
                {
                    "problem_id": prob["problem_id"],
                    "question": prob["question"],
                    "ground_truth": prob["answer"],
                    "generations": generations,
                    "extracted_answers": extracted,
                    "correctness": correctness,
                    "response_lengths": response_lengths,
                }
            )

        # acc@n = mean over problems of mean correctness across n samples (== overall pass rate)
        flat = [c for r in results for c in r["correctness"]]
        acc_at_n = sum(flat) / len(flat) if flat else 0.0
        any_correct_rate = n_correct_problems / len(results) if results else 0.0
        print(
            f"  acc@{sampling_params.n} (mean): {acc_at_n:.3f}    "
            f"any-correct rate: {any_correct_rate:.3f}    "
            f"({n_correct_problems}/{len(results)})"
        )

        out_path = results_dir / f"{label}_{bench_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "label": label,
                    "model": model_path,
                    "adapter": adapter_path,
                    "benchmark": bench_name,
                    "n_sampling": sampling_params.n,
                    "acc_mean": acc_at_n,
                    "any_correct_rate": any_correct_rate,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  wrote {out_path}")

    # Free GPU memory before loading the next model.
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
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_sampling", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=38912)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_model_len", type=int, default=40960,
                        help="vLLM context window (must be >= prompt + max_tokens).")
    parser.add_argument("--tensor_parallel_size", type=int,
                        default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--benchmarks", default="aime24,aime25",
                        help="Comma-separated subset of {aime24,aime25}")
    parser.add_argument("--models", default="baseline,lora,pretrained",
                        help="Comma-separated subset of {baseline,lora,pretrained}")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                        help="Inference backend: vllm (default) or hf (Transformers).")
    parser.add_argument("--max_questions", type=int, default=0,
                        help="If > 0, evaluate only the first N questions from each benchmark.")
    args = parser.parse_args()

    print("Loading benchmarks...")
    bench_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    benchmarks = {name: load_aime_benchmark(name) for name in bench_names}
    if args.max_questions and args.max_questions > 0:
        benchmarks = {
            name: problems[:args.max_questions]
            for name, problems in benchmarks.items()
        }
        for name, problems in benchmarks.items():
            print(f"  subset enabled: {name} -> {len(problems)} questions")

    if args.backend == "vllm":
        if SamplingParams is None:
            raise RuntimeError("vLLM backend requested but vllm is not installed in this environment.")
        sampling_params = SamplingParams(
            n=args.n_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
    else:
        sampling_params = SimpleNamespace(
            n=args.n_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    requested = {m.strip() for m in args.models.split(",") if m.strip()}

    if "baseline" in requested:
        run_one_model(
            label="baseline_sdpo_think_step100",
            model_path=BASE_MODEL,
            adapter_path=None,
            benchmarks=benchmarks,
            sampling_params=sampling_params,
            results_dir=results_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_thinking=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            backend=args.backend,
        )

    if "lora" in requested:
        run_one_model(
            label="lora_sdpo_plus_limo",
            model_path=BASE_MODEL,
            adapter_path=args.adapter_path,
            benchmarks=benchmarks,
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
            benchmarks=benchmarks,
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
