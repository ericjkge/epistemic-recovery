#!/usr/bin/env python3
"""Evaluate Kim et al. SDPO Qwen3 on GAIR/LIMO-v2 with vLLM.

This is intentionally self-contained research code. It mirrors the JSONL output
shape used by eval/eval.py and includes a local copy of its s1-style wait
injection helper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


BASE_MODEL = "beanie00/math-SDPO-Qwen3-8B-think-step-100"
BENCHMARK = "limov2"
DATA_SOURCE = "GAIR/LIMO-v2"
USER_TEMPLATE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# TODO: Replace the hard-coded placeholder with actual few-shot solutions.
FEWSHOT_PREFIX = """Below are examples of a solution. In this way, you can express uncertainty by using phrases such as "hmm" or "wait."

[HARD-CODED SECTION TO BE ADDED]

Now solve the next problem."""

BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class MockOutput:
    text: str


@dataclass
class MockCompletion:
    outputs: List[MockOutput]


def generate_with_wait_injection(
    llm,
    prompt_batch,
    sampling_params,
    wait_injections,
    wait_string="Wait",
    tokenizer=None,
):
    """Budget forcing / wait injection (s1-style).

    Generates each (prompt, sample) independently. For each injection round,
    generation stops at </think>; the wait_string is appended and generation
    continues. After all injections, a final 2048 token generation completes
    the response.

    Returns a list of MockCompletion objects with the same interface as vLLM
    completions (completion[i].outputs[j].text).
    """
    n = sampling_params.n

    if wait_injections == 0:
        return llm.generate(prompt_batch, sampling_params)

    if tokenizer is None:
        raise ValueError("tokenizer is required when wait injections are enabled")

    # Flatten so every (prompt, sample) is an independent sequence.
    flat_prompts = [p for p in prompt_batch for _ in range(n)]
    accumulated = [""] * len(flat_prompts)
    thinking_token_budget = 36864
    final_answer_token_budget = 2048
    remaining_thinking_tokens = [thinking_token_budget] * len(flat_prompts)
    wait_suffix = f"{wait_string}\n"
    wait_suffix_tokens = len(tokenizer.encode(wait_suffix, add_special_tokens=False))

    def generate_active_samples(current_prompts, token_budgets, stop_at_think):
        active_indices = [i for i, budget in enumerate(token_budgets) if budget > 0]
        if not active_indices:
            return {}

        params = []
        for i in active_indices:
            params_kwargs = {
                "temperature": sampling_params.temperature,
                "max_tokens": token_budgets[i],
                "n": 1,
                "top_p": sampling_params.top_p,
            }
            if stop_at_think:
                params_kwargs["stop"] = ["</think>"]
                params_kwargs["include_stop_str_in_output"] = False
            params.append(SamplingParams(**params_kwargs))

        completions = llm.generate([current_prompts[i] for i in active_indices], params)
        return {active_indices[i]: completions[i].outputs[0] for i in range(len(active_indices))}

    generated_outputs = generate_active_samples(flat_prompts, remaining_thinking_tokens, stop_at_think=True)
    for i, generated_output in generated_outputs.items():
        accumulated[i] += generated_output.text
        remaining_thinking_tokens[i] = max(0, remaining_thinking_tokens[i] - len(generated_output.token_ids))

    for injection_idx in range(wait_injections):
        round_budgets = [0] * len(flat_prompts)
        for i in range(len(flat_prompts)):
            if remaining_thinking_tokens[i] <= 0:
                continue
            if remaining_thinking_tokens[i] <= wait_suffix_tokens:
                print(
                    f"Wait round {injection_idx + 1}: stopping extra thinking for "
                    f"prompt_idx={i // n}, sample_idx={i % n}, "
                    f"remaining_thinking_tokens={remaining_thinking_tokens[i]}"
                )
                remaining_thinking_tokens[i] = 0
                continue
            accumulated[i] += wait_suffix
            remaining_thinking_tokens[i] -= wait_suffix_tokens
            round_budgets[i] = remaining_thinking_tokens[i]

        current_prompts = [flat_prompts[i] + accumulated[i] for i in range(len(flat_prompts))]
        generated_outputs = generate_active_samples(current_prompts, round_budgets, stop_at_think=True)
        for i, generated_output in generated_outputs.items():
            accumulated[i] += generated_output.text
            remaining_thinking_tokens[i] = max(0, remaining_thinking_tokens[i] - len(generated_output.token_ids))

    for i in range(len(flat_prompts)):
        accumulated[i] += "</think>\n\n"

    final_prompts = [flat_prompts[i] + accumulated[i] for i in range(len(flat_prompts))]
    final_answer_budgets = [final_answer_token_budget] * len(flat_prompts)
    generated_outputs = generate_active_samples(final_prompts, final_answer_budgets, stop_at_think=False)

    full_texts = [
        accumulated[i] + (generated_outputs[i].text if i in generated_outputs else "")
        for i in range(len(flat_prompts))
    ]

    # Reshape back to (len(prompt_batch), n)
    results = []
    for i in range(len(prompt_batch)):
        outputs = [MockOutput(text=full_texts[i * n + j]) for j in range(n)]
        results.append(MockCompletion(outputs=outputs))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAIR/LIMO-v2 with vLLM.")
    parser.add_argument("--model_name_or_path", default=BASE_MODEL)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_sampling", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=38912) # No. of tokens allowed for response (ignored in wait-injection)
    parser.add_argument("--max_model_len", type=int, default=40960) # No. of tokens allowed for context window
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--modes", default="baseline,fewshot,wait")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wait_injections", type=int, default=1)
    parser.add_argument("--wait_string", default="Wait")
    parser.add_argument("--disable_thinking", action="store_true")
    return parser.parse_args()


def load_limov2(max_questions: int = 0) -> list[dict[str, Any]]:
    ds = load_dataset(DATA_SOURCE, split="train")
    if max_questions and max_questions > 0:
        ds = ds.select(range(min(max_questions, len(ds))))

    problems = []
    for i, row in enumerate(ds):
        question = row.get("question")
        answer = row.get("answer")
        if question is None or answer is None:
            continue
        problems.append(
            {
                "problem_id": f"limov2_{i:04d}",
                "question": str(question),
                "answer": str(answer).strip(),
                "extra_info": {
                    "index": i,
                    "dataset": DATA_SOURCE,
                    "split": "train",
                },
            }
        )
    return problems


def apply_chat_template(tokenizer, user_content: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_content}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_user_prompt(question: str, mode: str) -> str:
    prompt = USER_TEMPLATE.format(question=question)
    if mode == "fewshot" and FEWSHOT_PREFIX:
        return f"{FEWSHOT_PREFIX.rstrip()}\n\n{prompt}"
    return prompt


def build_prompts(tokenizer, problems: list[dict[str, Any]], mode: str, enable_thinking: bool) -> list[str]:
    return [
        apply_chat_template(tokenizer, build_user_prompt(problem["question"], mode), enable_thinking)
        for problem in problems
    ]


def extract_last_boxed(text: str) -> str:
    last = None
    for match in BOXED_RE.finditer(text):
        last = match
    return last.group(1).strip() if last else ""


def normalize_int(text: str) -> int | None:
    if text is None:
        return None
    s = str(text).replace("\\,", "").replace(",", "").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.strip("$ \t").lstrip("+")
    match = re.search(r"-?\d+", s)
    if not match:
        return None
    try:
        return int(match.group())
    except ValueError:
        return None


def grade_int(extracted: str, ground_truth: str) -> bool:
    pred = normalize_int(extracted)
    gold = normalize_int(ground_truth)
    return pred is not None and gold is not None and pred == gold


def split_thinking(response: str) -> tuple[str, str]:
    match = THINK_RE.search(response)
    if not match:
        return "", response
    thinking = match.group(1).strip()
    answer = response[match.end():].strip()
    return thinking, answer


def output_token_length(output: Any, tokenizer) -> int:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return len(tokenizer.encode(output.text, add_special_tokens=False))


def safe_model_label(model_name_or_path: str) -> str:
    label = model_name_or_path.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_").lower()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    tmp_path.replace(path)


def run_mode(
    mode: str,
    llm,
    tokenizer,
    problems: list[dict[str, Any]],
    sampling_params: SamplingParams,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    print(f"\n[{mode}] building prompts for {len(problems)} LIMO-v2 problems...")
    prompts = build_prompts(tokenizer, problems, mode, enable_thinking=not args.disable_thinking)
    print(f"[{mode}] submitting {len(prompts)} prompts to vLLM (n={sampling_params.n})...")

    if mode == "wait":
        completions = generate_with_wait_injection(
            llm,
            prompts,
            sampling_params,
            wait_injections=args.wait_injections,
            wait_string=args.wait_string,
            tokenizer=tokenizer,
        )
    else:
        completions = llm.generate(prompts, sampling_params)

    records = []
    correct_cnt = 0
    avg_acc_list = []
    total_response_tokens = 0
    total_responses = 0

    for problem, completion in tqdm(
        zip(problems, completions),
        total=len(problems),
        desc=f"[{mode}] grading",
        unit="problem",
    ):
        generated_responses = [output.text for output in completion.outputs]
        response_token_lengths = [output_token_length(output, tokenizer) for output in completion.outputs]
        thinking_answer_pairs = [split_thinking(response) for response in generated_responses]
        thinking_contents = [pair[0] for pair in thinking_answer_pairs]
        answer_responses = [pair[1] for pair in thinking_answer_pairs]
        thinking_token_lengths = [
            len(tokenizer.encode(thinking, add_special_tokens=False)) if thinking else 0
            for thinking in thinking_contents
        ]

        generated_answers = [extract_last_boxed(response) for response in answer_responses]
        answers_correctness = [grade_int(answer, problem["answer"]) for answer in generated_answers]
        is_correct = any(answers_correctness)
        if is_correct:
            correct_cnt += 1
        if answers_correctness:
            avg_acc_list.append(sum(answers_correctness) / len(answers_correctness))

        total_response_tokens += sum(response_token_lengths)
        total_responses += len(response_token_lengths)

        records.append(
            {
                "question": build_user_prompt(problem["question"], mode),
                "generated_responses": generated_responses,
                "answer_responses": answer_responses,
                "response_token_lengths": response_token_lengths,
                "thinking_contents": thinking_contents,
                "thinking_token_lengths": thinking_token_lengths,
                "extra_info": problem["extra_info"],
                "data_source": DATA_SOURCE,
                "generated_answers": generated_answers,
                "gold_answer": problem["answer"],
                "is_correct": is_correct,
                "answers_correctness": answers_correctness,
                "avg_response_token_length": (
                    sum(response_token_lengths) / len(response_token_lengths)
                    if response_token_lengths
                    else 0
                ),
                "avg_thinking_token_length": (
                    sum(thinking_token_lengths) / len(thinking_token_lengths)
                    if thinking_token_lengths
                    else 0
                ),
                "avg_accuracy": (
                    sum(answers_correctness) / len(answers_correctness)
                    if answers_correctness
                    else 0
                ),
            }
        )

    pass_at_n = correct_cnt / len(records) if records else 0
    avg_response_length = total_response_tokens / total_responses if total_responses else 0
    avg_at_n = sum(avg_acc_list) / len(avg_acc_list) if avg_acc_list else 0
    print(f"[{mode}] correct cnt / total cnt: {correct_cnt}/{len(records)}")
    print(f"[{mode}] Pass@{sampling_params.n}: {pass_at_n:.4f}")
    print(f"[{mode}] Average Response Length (tokens): {avg_response_length:.2f}")
    print(f"[{mode}] Avg@{sampling_params.n}: {avg_at_n:.4f}")
    return records


def main() -> None:
    args = parse_args()
    problems = load_limov2(args.max_questions)
    if not problems:
        raise RuntimeError("No LIMO-v2 problems loaded.")
    print(f"Loaded {len(problems)} problems from {DATA_SOURCE}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=args.n_sampling,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    requested_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    allowed_modes = {"baseline", "fewshot", "wait"}
    unknown_modes = sorted(set(requested_modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}. Expected one or more of {sorted(allowed_modes)}")

    results_dir = Path(args.results_dir)
    model_label = safe_model_label(args.model_name_or_path)
    for mode in requested_modes:
        records = run_mode(mode, llm, tokenizer, problems, sampling_params, args)
        out_path = results_dir / f"{model_label}_{BENCHMARK}_{mode}.jsonl"
        write_jsonl(out_path, records)
        print(f"[{mode}] wrote {out_path}")


if __name__ == "__main__":
    main()
