#!/usr/bin/env python3
"""Evaluate Kim et al. SDPO Qwen3 on GAIR/LIMO-v2 with vLLM.

This is intentionally self-contained research code. It mirrors the JSONL output
shape used by eval/eval.py.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
from math import isclose
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from latex2sympy2 import latex2sympy
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr
except Exception:  # pragma: no cover - keeps extraction usable if optional grader deps are absent.
    latex2sympy = None
    N = None
    simplify = None
    parse_latex = None
    parse_expr = None


BASE_MODEL = "beanie00/math-SDPO-Qwen3-8B-think-step-100"
BENCHMARK = "limov2"
DATA_SOURCE = "GAIR/LIMO-v2"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
USER_TEMPLATE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# TODO: Replace the hard-coded placeholder with actual few-shot solutions.
FEWSHOT_PREFIX = """Below are examples of a solution. In this way, you can express uncertainty by using phrases such as "hmm" or "wait."

[HARD-CODED SECTION TO BE ADDED]

Now solve the next problem."""

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAIR/LIMO-v2 with vLLM.")
    parser.add_argument("--model_name_or_path", default=BASE_MODEL)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n_sampling", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=38912) # No. of tokens allowed for response
    parser.add_argument("--max_model_len", type=int, default=40960) # No. of tokens allowed for context window
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_questions", type=int, default=0)
    parser.add_argument("--modes", default="baseline,fewshot")
    parser.add_argument("--seed", type=int, default=0)
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
    marker = r"\boxed{"
    last = ""
    search_start = 0

    while True:
        start = text.find(marker, search_start)
        if start == -1:
            return last.strip()

        content_start = start + len(marker)
        depth = 1
        i = content_start
        while i < len(text) and depth > 0:
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            i += 1

        if depth == 0:
            last = text[content_start : i - 1]
            search_start = i
        else:
            search_start = content_start


def parse_digits(text: Any) -> float | None:
    s = str(text).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        if s.endswith("%"):
            try:
                return float(s[:-1].rstrip("\\")) / 100
            except ValueError:
                return None
    return None


def numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(prediction, reference, abs_tol=1e-4)


def strip_math_string(text: Any) -> str:
    """Small local copy of eval/utils/parser.py normalization for math answers."""
    s = str(text).strip().replace("\n", "").rstrip(".")
    s = s.replace("\\!", "")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        s = s.replace(key, "")
    s = s.replace("\\emptyset", "{}")
    s = s.replace("infinity", "\\infty").replace("+\\inity", "\\infty")
    if "\\infty" not in s:
        s = s.replace("inf", "\\infty")
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    return s.replace(" ", "")


def split_top_level_commas(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def _parse_symbolic(text: str) -> Any:
    parsers = [parse_latex, parse_expr, latex2sympy]
    for parser in parsers:
        if parser is None:
            continue
        for candidate in (text.replace("\\\\", "\\"), text):
            try:
                return parser(candidate)
            except Exception:
                pass
    return text


def symbolic_equal(prediction: str, reference: str) -> bool:
    if simplify is None or N is None:
        return False

    pred_expr = _parse_symbolic(prediction)
    ref_expr = _parse_symbolic(reference)

    try:
        if str(pred_expr) == str(ref_expr) or pred_expr == ref_expr:
            return True
    except Exception:
        pass

    try:
        if pred_expr.equals(ref_expr) or simplify(pred_expr - ref_expr) == 0:
            return True
    except Exception:
        pass

    try:
        if abs(pred_expr.lhs - pred_expr.rhs).equals(abs(ref_expr.lhs - ref_expr.rhs)):
            return True
    except Exception:
        pass

    try:
        return numeric_equal(float(N(pred_expr)), float(N(ref_expr)))
    except Exception:
        return False


def symbolic_equal_process(prediction: str, reference: str, output_queue: multiprocessing.Queue) -> None:
    output_queue.put(symbolic_equal(prediction, reference))


def call_with_timeout(func, *args, timeout: int = 3) -> bool:
    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=func, args=args + (output_queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return bool(output_queue.get()) if not output_queue.empty() else False


def math_equal(prediction: Any, reference: Any, timeout: bool = True, depth: int = 0) -> bool:
    """Local copy of eval/utils/grader.py's exact/numeric/symbolic equality pattern."""
    if depth > 5 or prediction is None or reference is None:
        return False

    prediction = strip_math_string(prediction)
    reference = strip_math_string(reference)
    if prediction.lower() == reference.lower():
        return True

    pred_num = parse_digits(prediction)
    ref_num = parse_digits(reference)
    if pred_num is not None and ref_num is not None:
        return numeric_equal(pred_num, ref_num)

    pred_parts = split_top_level_commas(prediction.strip("[]()"))
    ref_parts = split_top_level_commas(reference.strip("[]()"))
    if "," in prediction and "," in reference and len(pred_parts) == len(ref_parts):
        return all(math_equal(p, r, timeout=timeout, depth=depth + 1) for p, r in zip(pred_parts, ref_parts))

    pred_str = prediction
    ref_str = reference
    for char in ["{", "}", "(", ")"]:
        pred_str = pred_str.replace(char, "")
        ref_str = ref_str.replace(char, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        left_pred, right_pred = prediction.split("=")
        left_ref, right_ref = reference.split("=")
        prediction = f"{left_pred.strip()}-({right_pred.strip()})"
        reference = f"{left_ref.strip()}-({right_ref.strip()})"
    elif prediction.count("=") == 1 and "=" not in reference and len(prediction.split("=")[0]) <= 2:
        return math_equal(prediction.split("=")[1], reference, timeout=timeout, depth=depth + 1)
    elif reference.count("=") == 1 and "=" not in prediction and len(reference.split("=")[0]) <= 2:
        return math_equal(prediction, reference.split("=")[1], timeout=timeout, depth=depth + 1)

    if timeout:
        return call_with_timeout(symbolic_equal_process, prediction, reference)
    return symbolic_equal(prediction, reference)


def grade_answer(extracted: str, ground_truth: str) -> bool:
    return math_equal(extracted, ground_truth)


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
        answers_correctness = [grade_answer(answer, problem["answer"]) for answer in generated_answers]
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
    allowed_modes = {"baseline", "fewshot"}
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
