#!/usr/bin/env python3
"""Iterative-injection generator for on-policy reasoning traces.

For each seed problem:
  Round 1:
    - sample a response (Qwen3 thinking-mode chat template + boxed-answer system msg)
    - extract last \\boxed{...}, grade against integer GT
    - if correct → record, done
  Round k > 1:
    - take the prior round's text
    - strip everything from the LAST `</think>` (or after the last `\\boxed{}` if
      no `</think>`) so we re-enter thinking mode
    - append injection phrase " Wait, the answer is wrong. Let's think again.\\n\\n"
    - resample as a *continuation* of the assistant turn (vLLM raw-prompt mode)
    - grade; loop until correct OR max_rounds OR cumulative tokens > token_cap

Output: outputs/injection_traces.jsonl, one record per problem with all rounds and
final correctness. outputs/stats.json reports the round-by-round pass curve.

Backends: vLLM only — the same setup evaluate_aime.py uses. Designed to run on the
same H100 instance.

Run from sdpo_limo_llamafactory/ (or anywhere — paths are absolute):
    python3 experiments/experiment5/generate_injections.py \\
        --seeds experiments/experiment5/seed_problems.jsonl \\
        --output_dir experiments/experiment5/outputs \\
        --model lora                                # or 'baseline' / 'pretrained'
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

EXP5_DIR = Path(__file__).resolve().parent
PARENT_DIR = EXP5_DIR.parent.parent  # sdpo_limo_llamafactory/
sys.path.insert(0, str(PARENT_DIR / "eval"))
# Reuse the answer extractor + grader from evaluate_aime.py to stay consistent
# with the held-out benchmark grader.
from evaluate_aime import extract_last_boxed, grade_int  # type: ignore  # noqa: E402

BASE_MODEL = "beanie00/math-SDPO-Qwen3-8B-think-step-100"
PRETRAINED = "Qwen/Qwen3-8B"

USER_TEMPLATE = "{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
INJECTION_PHRASE = " Wait, the answer is wrong. Let's think again.\n\n"


# ── Trace surgery ────────────────────────────────────────────────────────────

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def reopen_thinking(prior: str) -> str:
    """Prepare a prior assistant response as a continuation prompt that re-enters
    thinking mode.

    Three trace shapes:
      (a) standard:  "<think>...thoughts...</think>\\n\\n...answer...\\boxed{X}..."
          → drop everything from the LAST </think> onward, so the assistant
            keeps writing inside thinking mode.
      (b) empty-think (LIMO LoRA pathology):
          "<think></think>\\n\\nthoughts...\\boxed{X}"
          → drop the empty <think></think> and put thoughts inside a fresh
            `<think>` block.
      (c) truncated, no </think>: keep prior verbatim — model already hasn't
          closed thinking, we just continue.

    Returns prior with the injection phrase appended at the end. The caller
    feeds this back through the chat template as a continuation.
    """
    last_close = prior.rfind(_THINK_CLOSE)
    if last_close != -1:
        # Case (a): preserve thoughts up to but not including </think>.
        # If the immediate preceding chars look like an empty <think></think>
        # (case b), promote the post-</think> reasoning back into the think block.
        last_open = prior.rfind(_THINK_OPEN, 0, last_close)
        gap = prior[last_open + len(_THINK_OPEN): last_close] if last_open != -1 else ""
        if last_open != -1 and len(gap.strip()) < 10:
            # Empty think block — pull the post-</think> reasoning back inside.
            tail_start = last_close + len(_THINK_CLOSE)
            tail = prior[tail_start:]
            # Drop the trailing \boxed{X} answer line — we'll regenerate that.
            box_idx = tail.rfind("\\boxed{")
            if box_idx > 0:
                line_start = tail.rfind("\n", 0, box_idx)
                tail = tail[: line_start if line_start != -1 else box_idx]
            reopened = prior[:last_open + len(_THINK_OPEN)] + "\n" + tail.rstrip()
        else:
            reopened = prior[:last_close].rstrip()
    elif _THINK_OPEN in prior:
        # Case (c): unclosed think — strip trailing partial-answer lines.
        reopened = prior.rstrip()
    else:
        # No think tag at all (shouldn't happen with Qwen3 chat template, but
        # tolerate it). Wrap retroactively.
        reopened = f"{_THINK_OPEN}\n{prior.rstrip()}"

    return reopened + INJECTION_PHRASE


# ── Prompting ────────────────────────────────────────────────────────────────

def render_initial_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def render_continuation_prompt(tokenizer, question: str, prior_with_injection: str) -> str:
    """Build a chat-template prompt that ends with an open assistant turn whose
    content already includes `prior_with_injection`. vLLM treats it as a
    continuation prompt — generated tokens become more assistant content.

    We avoid `continue_final_message=True` since support is uneven across HF
    template versions; appending raw text after `add_generation_prompt=True`
    gives the same effect deterministically for Qwen3 templates that end with
    `<|im_start|>assistant\\n`.
    """
    messages = [
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    # Older Qwen3 chat-template versions auto-injected `<think>\n` immediately
    # after `<|im_start|>assistant\n`; current versions don't (the model emits
    # `<think>` itself in thinking mode). This de-dup branch is harmlessly inert
    # on current templates but kept defensive for older transformers caches.
    if base.rstrip().endswith(_THINK_OPEN) and prior_with_injection.startswith(_THINK_OPEN):
        prior_with_injection = prior_with_injection[len(_THINK_OPEN):].lstrip("\n")
    return base + prior_with_injection


# ── vLLM driver ──────────────────────────────────────────────────────────────

def load_model(model_path: str, adapter_path: Optional[str], max_model_len: int,
               gpu_mem_util: float, tensor_parallel: int):
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enforce_eager=False,
    )
    lora_request = None
    if adapter_path:
        kwargs.update(enable_lora=True, max_lora_rank=64, max_loras=1)
        lora_request = LoRARequest("exp5_policy", 1, adapter_path)
    return LLM(**kwargs), lora_request


def generate_batch(llm, lora_request, prompts: list[str], sampling_params):
    """Submit all prompts in a single vLLM call so the scheduler can pack them
    onto the GPU concurrently. Returns a list of (text, n_tokens) in input order.
    Single-call batching is the whole point of using vLLM — calling once per
    prompt throws away ~10x throughput on H100."""
    gen_kwargs = {}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request
    outs = llm.generate(prompts, sampling_params, use_tqdm=False, **gen_kwargs)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(EXP5_DIR / "seed_problems.jsonl"))
    ap.add_argument("--output_dir", default=str(EXP5_DIR / "outputs"))
    ap.add_argument("--model", default="lora",
                    choices=("baseline", "lora", "pretrained"),
                    help="Generation policy. 'lora' uses --adapter / --merged "
                         "(typically experiment 4's winner). 'baseline' uses raw SDPO. "
                         "'pretrained' uses Qwen/Qwen3-8B.")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir for --model lora. Falls back to --merged "
                         "if both given and the merged checkpoint exists.")
    ap.add_argument("--merged", default=None,
                    help="Merged base+adapter checkpoint dir for --model lora "
                         "(workaround for vLLM LoRA crashes; same path as eval.sh's MERGED_DIR).")
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=10240,
                    help="Per-round max new tokens. 3 rounds × 10,240 = 30,720, sized to "
                         "match LIMO-v2's p99 (~31K) under the 32K SFT cutoff_len.")
    ap.add_argument("--cumulative_token_cap", type=int, default=30720,
                    help="Hard cap on total response length (sum of all rounds, generated "
                         "tokens only — excludes prompt and injection-phrase tokens). Once "
                         "exceeded, no further injections.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--frequency_penalty", type=float, default=0.0)
    ap.add_argument("--max_model_len", type=int, default=32768,
                    help="Qwen3-8B native context. Must hold prompt + accumulated prior "
                         "rounds + current round's max_tokens; at round 3 worst case this "
                         "is ~250 + 20,480 + 26 + 10,240 ≈ 31K, so 32K is the right size.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_problems", type=int, default=0,
                    help="If >0, only the first N seeds are processed.")
    ap.add_argument("--seed", type=int, default=42, help="vLLM sampling seed.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "injection_traces.jsonl"
    stats_path = out_dir / "stats.json"

    # ── Resolve model path ──────────────────────────────────────────────────
    adapter_path: Optional[str] = None
    if args.model == "baseline":
        model_path = BASE_MODEL
    elif args.model == "pretrained":
        model_path = PRETRAINED
    else:  # lora
        if args.merged and Path(args.merged, "config.json").exists():
            model_path = args.merged
            adapter_path = None
            print(f"  loading lora as merged checkpoint: {model_path}")
        elif args.adapter and Path(args.adapter).exists():
            model_path = BASE_MODEL
            adapter_path = args.adapter
            print(f"  loading base + LoRA adapter: {adapter_path}")
        else:
            raise SystemExit(
                "ERROR: --model lora requires --merged (preferred) or --adapter pointing "
                "at a valid checkpoint. (Tip: pass experiment 4's winning adapter/merged dir.)"
            )

    # ── Load seed problems ─────────────────────────────────────────────────
    seeds = []
    with open(args.seeds) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    if args.max_problems > 0:
        seeds = seeds[: args.max_problems]
    print(f"  seeds: {len(seeds)} from {args.seeds}")

    # ── Load tokenizer + model ─────────────────────────────────────────────
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  loading model: {model_path}")
    llm, lora_request = load_model(
        model_path, adapter_path,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_memory_utilization,
        tensor_parallel=args.tensor_parallel_size,
    )

    sp = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
    )

    # ── Per-round, batched driver ──────────────────────────────────────────
    # Outer loop is over rounds, not problems: every active problem's prompt
    # for round k is submitted to vLLM in a single batched call. Problems that
    # answer correctly (or hit the cumulative-token cap) drop out before
    # round k+1, so subsequent rounds only spend compute on the still-failing
    # subset. This is the throughput shape vLLM is designed for.
    state: dict[str, dict] = {
        prob["problem_id"]: {
            "prob": prob,
            "gt": int(prob["answer"]),
            "rounds": [],
            "cumulative_tokens": 0,
            "final_correct": False,
            "current_text": "",
            "reopened": "",          # set on rounds k>0; needed to reconstruct full trace
            "done": False,
        }
        for prob in seeds
    }
    round_correct = [0] * args.max_rounds  # cumulative correct after round k
    t0 = time.time()

    for round_idx in range(args.max_rounds):
        active_ids = [pid for pid, s in state.items() if not s["done"]]
        if not active_ids:
            break

        # Build prompts for all active problems
        prompts = []
        for pid in active_ids:
            s = state[pid]
            question = s["prob"]["question"]
            if round_idx == 0:
                prompts.append(render_initial_prompt(tokenizer, question))
            else:
                s["reopened"] = reopen_thinking(s["current_text"])
                prompts.append(render_continuation_prompt(tokenizer, question, s["reopened"]))

        t_round = time.time()
        print(
            f"  [round {round_idx + 1}/{args.max_rounds}]  submitting "
            f"{len(prompts)} prompts to vLLM…",
            flush=True,
        )
        results = generate_batch(llm, lora_request, prompts, sp)
        round_secs = time.time() - t_round

        # Process results, mark problems done as appropriate
        n_correct_this_round = 0
        n_capped_this_round = 0
        for pid, (gen_text, gen_tokens) in zip(active_ids, results):
            s = state[pid]
            # Reconstruct full assistant text. vLLM only returns the new tokens;
            # for continuation prompts (round > 0) we prepend the reopened
            # prior so the trace contains the whole multi-round thinking.
            if round_idx == 0:
                s["current_text"] = gen_text
            else:
                s["current_text"] = s["reopened"] + gen_text

            s["cumulative_tokens"] += gen_tokens
            extracted = extract_last_boxed(s["current_text"])
            correct = grade_int(extracted, s["gt"])
            s["rounds"].append({
                "round": round_idx + 1,
                "new_tokens": gen_tokens,
                "extracted_answer": extracted,
                "correct": correct,
            })

            if correct:
                s["final_correct"] = True
                s["done"] = True
                n_correct_this_round += 1
                # cumulative-correct counter: this problem also "would have"
                # been correct at every later round k if we'd kept going.
                for k in range(round_idx, args.max_rounds):
                    round_correct[k] += 1
            elif s["cumulative_tokens"] >= args.cumulative_token_cap:
                s["done"] = True
                n_capped_this_round += 1

        elapsed = time.time() - t0
        print(
            f"    → +{n_correct_this_round} correct, +{n_capped_this_round} hit token cap, "
            f"{sum(1 for s in state.values() if not s['done'])} still active.  "
            f"round_secs={round_secs:.1f}s  total={elapsed:.0f}s",
            flush=True,
        )

    # Write traces in original seed order
    with open(traces_path, "w") as outf:
        for prob in seeds:
            s = state[prob["problem_id"]]
            outf.write(json.dumps({
                "problem_id": prob["problem_id"],
                "question": prob["question"],
                "ground_truth": s["gt"],
                "rounds": s["rounds"],
                "final_correct": s["final_correct"],
                "final_round": s["rounds"][-1]["round"] if s["rounds"] else 0,
                "cumulative_tokens": s["cumulative_tokens"],
                "final_text": s["current_text"],
            }, ensure_ascii=False) + "\n")

    n_processed = len(seeds)

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = {
        "n_problems": n_processed,
        "max_rounds": args.max_rounds,
        "round_pass_rate": [round_correct[k] / n_processed for k in range(args.max_rounds)] if n_processed else [],
        "round_correct": round_correct,
        "model": args.model,
        "adapter": adapter_path,
        "merged": args.merged if args.model == "lora" else None,
        "model_path": model_path,
        "args": vars(args),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  → {traces_path}")
    print(f"  → {stats_path}")
    print(f"  pass-rate by round: {stats['round_pass_rate']}")


if __name__ == "__main__":
    main()
