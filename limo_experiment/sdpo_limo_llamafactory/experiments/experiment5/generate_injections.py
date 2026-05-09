#!/usr/bin/env python3
"""Iterative-injection generator for on-policy reasoning traces (pass@K variant).

For each seed problem, run K independent injection chains in parallel:
  Round 1:
    - sample K responses from the same initial prompt (vLLM n=K)
    - for each chain: extract last \\boxed{...}, grade against integer GT
    - chains that grade correct are done; the others advance to round 2
  Round k > 1:
    - for each *still-active* (problem, chain) pair, take that chain's prior
      text, strip everything from the LAST `</think>` (or after the last
      `\\boxed{}` if no `</think>`) so we re-enter thinking mode, append the
      injection phrase, and resample as a continuation (n=1, raw-prompt mode)
    - grade; chain done if correct OR cumulative tokens > token_cap

Pass@K at round k = fraction of problems where ANY chain is correct by round k
(matches `evaluate_aime.py`'s any_correct_rate semantics).

Output: outputs/injection_traces.jsonl, one record per problem with all K chains'
rounds and final correctness. outputs/stats.json reports the round-by-round
pass@K curve and the per-chain pass@1 average for context.

Backends: vLLM only — same setup as evaluate_aime.py.

Run from sdpo_limo_llamafactory/ (or anywhere — paths are absolute):
    python3 experiments/experiment5/generate_injections.py \\
        --seeds experiments/experiment5/seed_problems.jsonl \\
        --output_dir experiments/experiment5/outputs \\
        --num_chains 4 \\
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
INJECTION_PHRASE = " Wait, that might be wrong. Let's try some different approaches.\n\n"
# Hard cap on the sum of *generated* tokens across rounds for one chain
# (excludes prompt and injection-phrase tokens). Sized so that when split
# 16K (R1) + 14720 (R2) the total still fits comfortably under
# qwen3_sdpo_lora_sft.yaml's cutoff_len=32768. Not a CLI flag — changing it
# means rethinking the per-round defaults too.
CUMULATIVE_TOKEN_CAP = 30720
# The phrase actually used by the current run; --injection_phrase overrides
# INJECTION_PHRASE without mutating the importable default. reopen_thinking
# reads this at call time so a CLI override propagates without plumbing.
_ACTIVE_INJECTION_PHRASE = INJECTION_PHRASE


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

    return reopened + _ACTIVE_INJECTION_PHRASE


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
    onto the GPU concurrently. Returns the raw RequestOutput list in input
    order — caller decides whether to read one or many outputs per prompt.
    Single-call batching is the whole point of using vLLM — calling once per
    prompt throws away ~10x throughput on H100."""
    gen_kwargs = {}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request
    return llm.generate(prompts, sampling_params, use_tqdm=False, **gen_kwargs)


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
    ap.add_argument("--num_chains", type=int, default=4,
                    help="Number of independent injection chains per problem (pass@K). "
                         "Round 1 samples this many candidates per problem in one vLLM call "
                         "(n=K); rounds 2+ each chain independently inject-and-retries.")
    ap.add_argument("--max_rounds", type=int, default=2)
    ap.add_argument("--max_tokens_r1", type=int, default=20000,
                    help="Max new tokens for round 1. Default 20000 covers baseline "
                         "SDPO's reasoning-length p99 (~9.6K observed under the old 10K "
                         "cap, true p99 is higher) plus the long tail seen on hard "
                         "problems where the model occasionally fills 16-18K and still "
                         "hasn't committed an answer. R1's prompt is just the question, "
                         "so the full vLLM context is available for generation.")
    ap.add_argument("--max_tokens_r2", type=int, default=10720,
                    help="Max new tokens for each continuation round (R2+). Default "
                         "10720 = 30720 (CUMULATIVE_TOKEN_CAP) - 20000 (R1), so a chain "
                         "that uses its full R1 budget can still use its full R2 budget "
                         "before the cumulative cap kicks in. Asymmetric vs R1 because "
                         "R2 has the R1 trace as prompt (less work to do) AND the R1 "
                         "trace eats into the max_model_len budget (less room available).")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--frequency_penalty", type=float, default=0.0)
    ap.add_argument("--max_model_len", type=int, default=32768,
                    help="Qwen3-8B native context. Must hold prompt + accumulated prior "
                         "rounds + current round's max_tokens; at round 2 worst case this "
                         "is ~250 (chat template + question) + 20,000 (R1 thinking) + "
                         "16 (injection) + 10,720 (R2) ≈ 31K, so 32K is the right size.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_problems", type=int, default=0,
                    help="If >0, only the first N seeds are processed.")
    ap.add_argument("--seed", type=int, default=42, help="vLLM sampling seed.")
    ap.add_argument("--injection_phrase", default=None,
                    help="Override the default injection phrase. Useful for ablations. "
                         "Pass the exact string including any leading space and trailing newlines "
                         "you want appended after the prior thinking.")
    args = ap.parse_args()

    global _ACTIVE_INJECTION_PHRASE
    if args.injection_phrase is not None:
        _ACTIVE_INJECTION_PHRASE = args.injection_phrase
    print(f"  injection_phrase: {_ACTIVE_INJECTION_PHRASE!r}")

    if args.num_chains < 1:
        raise SystemExit("--num_chains must be >= 1")

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
    K = args.num_chains
    print(f"  seeds: {len(seeds)} from {args.seeds}   chains/problem: {K}")

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

    # Round 1 uses n=K to fan out per problem; rounds 2+ use n=1 because each
    # chain has diverged into its own continuation prompt.
    sp_round1 = SamplingParams(
        n=K,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens_r1,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
    )
    sp_continuation = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens_r2,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
    )

    # ── Per-round, batched driver ──────────────────────────────────────────
    # Outer loop is over rounds, not (problem, chain) pairs: every active
    # chain's prompt for round k is submitted to vLLM in a single batched call.
    # Chains that answer correctly (or hit the cumulative-token cap) drop out
    # before round k+1, so subsequent rounds only spend compute on the
    # still-failing subset. This is the throughput shape vLLM is designed for.
    chains: dict[tuple[str, int], dict] = {}
    for prob in seeds:
        for c in range(K):
            chains[(prob["problem_id"], c)] = {
                "prob": prob,
                "gt": int(prob["answer"]),
                "chain_idx": c,
                "rounds": [],
                "cumulative_tokens": 0,
                "final_correct": False,
                "current_text": "",
                "reopened": "",
                "done": False,
            }

    # Pass@K tracking: round_correct[k] = number of problems where >=1 chain
    # is correct by the end of round k+1. Chain-level cumulative also tracked
    # for diagnosis (mean per-chain pass@1 per round).
    round_correct_passK = [0] * args.max_rounds
    round_correct_chains = [0] * args.max_rounds
    t0 = time.time()

    for round_idx in range(args.max_rounds):
        if round_idx == 0:
            # Round 1: one prompt per problem, n=K outputs each.
            active_problems = seeds
            prompts = [
                render_initial_prompt(tokenizer, prob["question"])
                for prob in active_problems
            ]
            t_round = time.time()
            print(
                f"  [round {round_idx + 1}/{args.max_rounds}]  submitting "
                f"{len(prompts)} prompts × n={K} = {len(prompts) * K} chains to vLLM…",
                flush=True,
            )
            outs = generate_batch(llm, lora_request, prompts, sp_round1)
            round_secs = time.time() - t_round

            n_chains_correct = 0
            n_chains_capped = 0
            for prob, comp in zip(active_problems, outs):
                # vLLM may not preserve order across the K outputs; treat them
                # as an unordered set (chain identity is post-hoc — assign in
                # whatever order vLLM returns them).
                for c, oc in enumerate(comp.outputs[:K]):
                    chain = chains[(prob["problem_id"], c)]
                    gen_text = oc.text
                    gen_tokens = len(oc.token_ids)
                    chain["current_text"] = gen_text
                    chain["cumulative_tokens"] += gen_tokens
                    extracted = extract_last_boxed(gen_text)
                    correct = grade_int(extracted, chain["gt"])
                    chain["rounds"].append({
                        "round": round_idx + 1,
                        "new_tokens": gen_tokens,
                        "extracted_answer": extracted,
                        "correct": correct,
                    })
                    if correct:
                        chain["final_correct"] = True
                        chain["done"] = True
                        n_chains_correct += 1
                    elif chain["cumulative_tokens"] >= CUMULATIVE_TOKEN_CAP:
                        chain["done"] = True
                        n_chains_capped += 1
        else:
            # Rounds 2+: one prompt per active chain (n=1).
            active_keys = [k for k, s in chains.items() if not s["done"]]
            if not active_keys:
                # Backfill remaining round counters with the current cumulative
                # state — pass@K and per-chain pass@1 plateau once everything
                # is done, but the vector should still have max_rounds entries.
                for k in range(round_idx, args.max_rounds):
                    round_correct_passK[k] = round_correct_passK[round_idx - 1]
                    round_correct_chains[k] = round_correct_chains[round_idx - 1]
                break

            prompts = []
            for key in active_keys:
                s = chains[key]
                s["reopened"] = reopen_thinking(s["current_text"])
                question = s["prob"]["question"]
                prompts.append(render_continuation_prompt(tokenizer, question, s["reopened"]))

            t_round = time.time()
            print(
                f"  [round {round_idx + 1}/{args.max_rounds}]  submitting "
                f"{len(prompts)} active-chain prompts to vLLM…",
                flush=True,
            )
            outs = generate_batch(llm, lora_request, prompts, sp_continuation)
            round_secs = time.time() - t_round

            n_chains_correct = 0
            n_chains_capped = 0
            for key, comp in zip(active_keys, outs):
                s = chains[key]
                oc = comp.outputs[0]
                gen_text = oc.text
                gen_tokens = len(oc.token_ids)
                # Reconstruct full assistant text for this chain. vLLM only
                # returns the new tokens; for continuation prompts we prepend
                # the reopened prior so the trace contains the whole multi-
                # round thinking.
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
                    n_chains_correct += 1
                elif s["cumulative_tokens"] >= CUMULATIVE_TOKEN_CAP:
                    s["done"] = True
                    n_chains_capped += 1

        # End-of-round accounting
        round_correct_chains[round_idx] = sum(
            1 for s in chains.values() if s["final_correct"]
        )
        round_correct_passK[round_idx] = sum(
            1
            for prob in seeds
            if any(chains[(prob["problem_id"], c)]["final_correct"] for c in range(K))
        )
        active_chains_remaining = sum(1 for s in chains.values() if not s["done"])
        elapsed = time.time() - t0
        print(
            f"    → +{n_chains_correct} chains correct, +{n_chains_capped} hit token cap, "
            f"{active_chains_remaining} chains still active.  "
            f"pass@{K}={round_correct_passK[round_idx]}/{len(seeds)}  "
            f"chain pass@1 cum={round_correct_chains[round_idx]}/{len(seeds) * K}  "
            f"round_secs={round_secs:.1f}s  total={elapsed:.0f}s",
            flush=True,
        )

    # Write traces in original seed order; one record per problem with K chains.
    with open(traces_path, "w") as outf:
        for prob in seeds:
            chain_records = []
            for c in range(K):
                s = chains[(prob["problem_id"], c)]
                chain_records.append({
                    "chain_idx": c,
                    "rounds": s["rounds"],
                    "final_correct": s["final_correct"],
                    "final_round": s["rounds"][-1]["round"] if s["rounds"] else 0,
                    "cumulative_tokens": s["cumulative_tokens"],
                    "final_text": s["current_text"],
                })
            any_correct = any(cr["final_correct"] for cr in chain_records)
            min_correct_round = min(
                (cr["final_round"] for cr in chain_records if cr["final_correct"]),
                default=0,
            )
            outf.write(json.dumps({
                "problem_id": prob["problem_id"],
                "question": prob["question"],
                "ground_truth": int(prob["answer"]),
                "num_chains": K,
                "chains": chain_records,
                "any_correct": any_correct,
                "min_correct_round": min_correct_round,
            }, ensure_ascii=False) + "\n")

    n_processed = len(seeds)
    n_chains_total = n_processed * K

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = {
        "n_problems": n_processed,
        "num_chains": K,
        "max_rounds": args.max_rounds,
        f"pass_at_{K}_round_rate": [
            round_correct_passK[k] / n_processed for k in range(args.max_rounds)
        ] if n_processed else [],
        f"pass_at_{K}_round_correct": round_correct_passK,
        "chain_pass_at_1_round_rate": [
            round_correct_chains[k] / n_chains_total for k in range(args.max_rounds)
        ] if n_chains_total else [],
        "chain_pass_at_1_round_correct": round_correct_chains,
        "model": args.model,
        "adapter": adapter_path,
        "merged": args.merged if args.model == "lora" else None,
        "model_path": model_path,
        "injection_phrase": _ACTIVE_INJECTION_PHRASE,
        "cumulative_token_cap": CUMULATIVE_TOKEN_CAP,
        "args": vars(args),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  → {traces_path}")
    print(f"  → {stats_path}")
    print(f"  pass@{K} by round: {stats[f'pass_at_{K}_round_rate']}")
    print(f"  per-chain pass@1 by round: {stats['chain_pass_at_1_round_rate']}")


if __name__ == "__main__":
    main()
