#!/usr/bin/env python3
"""Experiment 7 sanity eval — focused 7-problem AIME25 subset.

Problems:
  - aime25_000..aime25_004 (Problems 1-5): easy probes. Baseline competence
    check — if the recovery LoRA breaks problems the SDPO baseline already
    solves, we've over-fit to the recovery pattern.
  - aime25_017 (Problem 17, ans=82): 2x2 grid edge 2-red/2-blue colorings.
    Case 2 from experiments/experiment1/eval_results/epistemic_case_studies.md.
  - aime25_023 (Problem 23, ans=149): sin(7*pi*sin(5x)) zeros + tangents.
    Case 1 from the same file. Endpoint-exclusion off-by-one.

On both case studies the original LIMO LoRA reliably recovered to the
correct answer while the SDPO baseline collapsed to a clean-but-wrong
shortcut. Question for exp7: do 204 rows of pure recovery traces
transfer the same recovery behavior?

Defaults: models=baseline,lora; n_sampling=4; ~7*4*2 = 56 generations.
"""
import argparse
import sys
from pathlib import Path

EXP7_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP7_DIR.parent.parent  # sdpo_limo_llamafactory/
sys.path.insert(0, str(PARENT_DIR / "eval"))

from evaluate_aime import (  # noqa: E402
    BASE_MODEL,
    PRETRAINED,
    load_aime_benchmark,
    run_one_model,
)

CASE_STUDY_IDS = ("aime25_017", "aime25_023")
EASY_PROBE_IDS = tuple(f"aime25_{i:03d}" for i in range(5))
DEFAULT_PROBLEM_IDS = EASY_PROBE_IDS + CASE_STUDY_IDS


def filter_benchmark(problems, ids):
    keep = list(ids)
    by_id = {p["problem_id"]: p for p in problems}
    missing = [i for i in keep if i not in by_id]
    if missing:
        raise ValueError(f"Problem ids not found in benchmark: {missing}")
    return [by_id[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--lora_merged_path", default="",
                    help="If a merged checkpoint exists, vLLM loads it as a "
                         "plain model instead of base+adapter.")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--problem_ids", default=",".join(DEFAULT_PROBLEM_IDS))
    ap.add_argument("--models", default="baseline,lora",
                    help="Comma-separated subset of {baseline,lora,pretrained}")
    ap.add_argument("--n_sampling", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=24576)
    ap.add_argument("--max_model_len", type=int, default=28672)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--frequency_penalty", type=float, default=0.0)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    from vllm import SamplingParams  # imported lazily so help text works without vllm

    ids = tuple(s.strip() for s in args.problem_ids.split(",") if s.strip())
    print(f"Sanity eval on {len(ids)} problems: {list(ids)}")

    print("Loading aime25 benchmark...")
    full = load_aime_benchmark("aime25")
    subset = filter_benchmark(full, ids)
    benchmarks = {"aime25": subset}
    for p in subset:
        print(f"  {p['problem_id']}  ans={p['answer']}  q={p['question'][:80]}...")

    sampling_params = SamplingParams(
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
            backend="vllm",
        )

    if "lora" in requested:
        merged = Path(args.lora_merged_path) if args.lora_merged_path else None
        if merged and merged.exists() and (merged / "config.json").exists():
            lora_model_path = str(merged)
            lora_adapter = None
            print(f"  loading lora as merged checkpoint: {lora_model_path}")
        else:
            lora_model_path = BASE_MODEL
            lora_adapter = args.adapter_path
        run_one_model(
            label="lora_sdpo_plus_recovery_exp7",
            model_path=lora_model_path,
            adapter_path=lora_adapter,
            benchmarks=benchmarks,
            sampling_params=sampling_params,
            results_dir=results_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_thinking=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            backend="vllm",
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
            backend="vllm",
        )


if __name__ == "__main__":
    main()
