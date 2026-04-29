#!/usr/bin/env python3
"""Evaluate three Qwen3-8B variants on AIME24 + AIME25 with vLLM.

Models:
  baseline          beanie00/math-SDPO-Qwen3-8B-think-step-100  (no adapter)
  lora              same base + LoRA adapter from saves/qwen3_sdpo_limo_lora/
  pretrained        Qwen/Qwen3-8B  (upper bound, full thinking-mode pretrain)

For each (model, benchmark) we sample n=16 completions per problem with the Kim et al.
generation settings, extract the LAST \\boxed{...} as the answer, and grade against the
integer ground truth. Per-problem results are saved to results/{model}_{benchmark}.json.

Run from the sdpo_limo_llamafactory/ parent directory:
    source .venv-eval/bin/activate
    python eval/evaluate_aime.py --adapter_path saves/qwen3_sdpo_limo_lora
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


BASE_MODEL = "beanie00/math-SDPO-Qwen3-8B-think-step-100"
PRETRAINED = "Qwen/Qwen3-8B"

USER_TEMPLATE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

# data/ lives one level up from this script (sdpo_limo_llamafactory/data/)
_LOCAL_PARQUET_DIR = Path(__file__).parent.parent / "data"


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


def _load_aime_parquet(name: str):
    """Load from the local parquet in data/{name}.parquet.

    Both aime24 and aime25 use the same logical schema but different encodings:
      - aime24: struct columns  (extra_info.raw_problem, reward_model.ground_truth)
      - aime25: JSON-string columns (need json.loads on extra_info / reward_model)

    Returns a list of {problem_id, question, answer:int} or None if file not found.
    """
    parquet_path = (_LOCAL_PARQUET_DIR / f"{name}.parquet").resolve()
    if not parquet_path.exists():
        return None

    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=str(parquet_path), split="train")

    out = []
    for i, row in enumerate(ds):
        # --- question ---
        extra = row.get("extra_info")
        if isinstance(extra, str):
            extra = json.loads(extra)
        question = extra.get("raw_problem") if isinstance(extra, dict) else None

        # Fallback: pull from prompt field (strip instruction suffix added at eval time)
        if not question:
            prompt = row.get("prompt")
            if isinstance(prompt, str):
                prompt = json.loads(prompt)
            if isinstance(prompt, list) and prompt:
                question = prompt[0].get("content", "")
                question = re.sub(r"\s*Please reason step by step.*$", "", question, flags=re.DOTALL).strip()

        # --- answer ---
        reward = row.get("reward_model")
        if isinstance(reward, str):
            reward = json.loads(reward)
        answer = reward.get("ground_truth") if isinstance(reward, dict) else None

        if not question or answer is None:
            continue
        try:
            a_int = int(str(answer).strip())
        except ValueError:
            continue
        out.append({"problem_id": f"{name}_{i:03d}", "question": question, "answer": a_int})

    return out or None


def load_aime_benchmark(name: str):
    """Return a list of {id, question, answer:int} dicts.

    Tries the local parquet (data/{name}.parquet) first, then falls back to HuggingFace.
    """
    local = _load_aime_parquet(name)
    if local:
        print(f"  loaded {name} from local parquet — {len(local)} problems")
        return local
    print(f"  local parquet not found for {name}, falling back to HuggingFace...")

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
        print(f"\n[{label} / {bench_name}] generating n={sampling_params.n} for {len(problems)} problems "
              f"(per-problem streaming)...")
        prompts = build_prompts(tokenizer, problems, enable_thinking=enable_thinking)

        if backend == "hf":
            if hasattr(model_runner, "hf_device_map"):
                hf_input_device = next(model_runner.parameters()).device
            else:
                hf_input_device = hf_device

        results = []
        n_correct_problems = 0
        running_correct = 0
        running_total = 0
        for i, (prob, prompt) in enumerate(zip(problems, prompts), start=1):
            if backend == "vllm":
                gen_kwargs = {}
                if lora_request is not None:
                    gen_kwargs["lora_request"] = lora_request
                comp = model_runner.generate([prompt], sampling_params, use_tqdm=False, **gen_kwargs)[0]
                generations = [o.text for o in comp.outputs]
                response_lengths = [len(o.token_ids) for o in comp.outputs]
            else:
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
                generations = []
                response_lengths = []
                for seq in generated:
                    seq = seq[input_len:]
                    generations.append(tokenizer.decode(seq, skip_special_tokens=True))
                    response_lengths.append(len(seq.tolist()))

            extracted = [extract_last_boxed(g) for g in generations]
            correctness = [grade_int(e, prob["answer"]) for e in extracted]
            if any(correctness):
                n_correct_problems += 1
            n_ok = sum(correctness)
            running_correct += n_ok
            running_total += len(correctness)
            mean_len = sum(response_lengths) // len(response_lengths)
            print(
                f"  [{i:>2}/{len(problems)}] "
                f"correct={n_ok}/{len(correctness)}  "
                f"running_acc={running_correct/running_total:.3f}  "
                f"gt={prob['answer']}  "
                f"extracted={extracted}  "
                f"mean_len={mean_len}",
                flush=True,
            )
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
    parser.add_argument("--lora_merged_path", default="saves/qwen3_sdpo_limo_lora_merged",
                        help="If this path exists, the 'lora' model loads it as a plain merged "
                             "checkpoint instead of base+adapter. Workaround for vLLM LoRA/Punica "
                             "crashes.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_sampling", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=24576,
                        help="Max new tokens per sample. 24576 covers >99%% of legitimate "
                             "AIME reasoning traces (Qwen3-8B baseline median ~6K, p95 ~25K). "
                             "Larger caps just give repetition loops more room to run.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--frequency_penalty", type=float, default=0.0,
                        help="vLLM frequency_penalty. Default 0 (off). Setting >0 penalizes "
                             "token reuse aggressively enough to derail Qwen3 reasoning "
                             "(e.g. spelling digits in English once they're suppressed).")
    parser.add_argument("--max_model_len", type=int, default=28672,
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
