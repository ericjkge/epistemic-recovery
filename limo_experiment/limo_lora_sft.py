"""
LoRA SFT on LIMO-v2 (GAIR/LIMO-v2)
=====================================
Fine-tunes beanie00/math-SDPO-Qwen3-8B-think-step-100 (or any local copy)
using LoRA via PEFT + TRL SFTTrainer on the LIMO-v2 reasoning traces.

The model is trained in Qwen3 thinking-mode format:
  <|im_start|>user
  {problem}\nSolve step by step …<|im_end|>
  <|im_start|>assistant
  <think>
  {solution / reasoning trace}
  </think>
  {final boxed answer}<|im_end|>

Usage:
  python limo_lora_sft.py \
      --base_model beanie00/math-SDPO-Qwen3-8B-think-step-100 \
      --output_dir ./lora_outputs/rank16 \
      --lora_rank 16 \
      --lora_alpha 32 \
      --num_train_epochs 2 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --learning_rate 5e-5

After training, use --merge_after_training to fuse LoRA weights into a standalone HF model.
Use --sweep for a grid search over lr × epochs.
Use --eval_only to run generation-based epistemic evaluation on a saved adapter.
"""

import argparse
import copy
import json
import os
import re
import sys

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ── LIMO-v2 column discovery ──────────────────────────────────────────────────
# GAIR/LIMO-v2 columns (as of April 2025):
#   - 'question' or 'problem': the math problem text
#   - 'solution': the long-form reasoning trace / chain-of-thought
#   - 'answer': the final numerical answer
POSSIBLE_PROBLEM_COLS = ["question", "problem", "input"]
POSSIBLE_SOLUTION_COLS = ["solution", "reasoning", "output", "cot"]
POSSIBLE_ANSWER_COLS = ["answer", "final_answer", "label"]

SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. "
    "Think through the problem carefully, then provide the final answer within \\boxed{}."
)

# ── Epistemic token list ──────────────────────────────────────────────────────
EPISTEMIC_TOKENS = [
    "wait", "hmm", "perhaps", "maybe", "actually",
    "alternatively", "seems", "might", "likely", "check",
    "let me", "reconsider", "hold on", "no,", "but wait",
]

# ── Hardcoded eval problems for generation-based tracking ────────────────────
EVAL_PROBLEMS = [
    "What is the sum of the first 100 positive integers?",
    "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
    "Solve for x: 3x² - 12x + 9 = 0",
    "A circle has area 49π. What is its circumference?",
    "How many ways can you arrange the letters in the word MATH?",
    "Find the derivative of f(x) = x³ - 4x² + 7x - 2.",
    "What is the probability of rolling a sum of 7 with two fair dice?",
    "Simplify: (2³ × 3²) / (6 × 4)",
]


# ── Column detection ──────────────────────────────────────────────────────────

def detect_column(ds_columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in ds_columns:
            return c
    return None


def extract_boxed_answer(text: str) -> str:
    """Pull the last \\boxed{…} from a solution string."""
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else text.strip()


def build_chat_text(tokenizer, problem: str, solution: str, answer: str) -> str:
    """
    Format one training example into the model's chat template.

    The assistant turn wraps the solution in <think>…</think> and appends
    a brief boxed-answer line.  If the solution already ends with \\boxed{},
    we leave the answer part as-is; otherwise we append it.
    """
    # Wrap thinking trace
    if solution.strip().startswith("<think>"):
        assistant_content = solution.strip()
    else:
        assistant_content = f"<think>\n{solution.strip()}\n</think>"

    # Append final answer if it isn't already present
    if "\\boxed{" not in assistant_content.split("</think>")[-1]:
        assistant_content += f"\n\nThe answer is $\\boxed{{{answer}}}$."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{problem}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        },
        {"role": "assistant", "content": assistant_content},
    ]

    # apply_chat_template with tokenize=False returns the full string
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback for tokenizers without chat template
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
        )
    return text


def load_and_format_limo(tokenizer, max_seq_length: int, split: str = "train") -> Dataset:
    """Download GAIR/LIMO-v2 and format every example as a full chat string."""
    print("Loading GAIR/LIMO-v2 from HuggingFace …")
    raw_ds = load_dataset("GAIR/LIMO-v2", split=split)
    cols = raw_ds.column_names
    print(f"  Columns found: {cols}")

    problem_col = detect_column(cols, POSSIBLE_PROBLEM_COLS)
    solution_col = detect_column(cols, POSSIBLE_SOLUTION_COLS)
    answer_col = detect_column(cols, POSSIBLE_ANSWER_COLS)

    if problem_col is None or solution_col is None:
        raise ValueError(
            f"Could not find problem/solution columns in LIMO-v2. "
            f"Available: {cols}. "
            f"Tried problem={POSSIBLE_PROBLEM_COLS}, solution={POSSIBLE_SOLUTION_COLS}"
        )

    print(f"  Using: problem='{problem_col}', solution='{solution_col}', answer='{answer_col}'")

    formatted_texts = []
    skipped = 0
    for ex in raw_ds:
        problem = str(ex[problem_col]).strip()
        solution = str(ex[solution_col]).strip()
        answer = str(ex[answer_col]).strip() if answer_col else extract_boxed_answer(solution)

        if not problem or not solution:
            skipped += 1
            continue

        text = build_chat_text(tokenizer, problem, solution, answer)

        # Filter by sequence length (crude token count to avoid OOM)
        # We use a rough heuristic: ~4 chars/token
        if len(text) // 4 > max_seq_length:
            skipped += 1
            continue

        formatted_texts.append({"text": text})

    if skipped:
        print(f"  Skipped {skipped} examples (empty or too long).")
    print(f"  Formatted {len(formatted_texts)} training examples.")

    return Dataset.from_list(formatted_texts)


def build_lora_config(args) -> LoraConfig:
    """
    Construct PEFT LoraConfig.

    Key parameters to sweep:
      lora_rank  (r): 8, 16, 32, 64  → higher = more capacity but more memory
      lora_alpha    : typically equal to rank (α/r=1) per Schulman et al. calibration
      target_modules: all attention projections + FFN gates for full coverage
    """
    target_modules = args.target_modules.split(",") if args.target_modules else [
        # Qwen3 attention projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Qwen3 FFN gates (SwiGLU)
        "gate_proj", "up_proj", "down_proj",
    ]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,
    )


# ── Epistemic token utilities ─────────────────────────────────────────────────

def count_epistemic_tokens(text: str) -> int:
    """Count occurrences of epistemic tokens (case-insensitive) in text."""
    text_lower = text.lower()
    return sum(text_lower.count(tok) for tok in EPISTEMIC_TOKENS)


def generate_responses(
    model,
    tokenizer,
    problems: list[str],
    max_new_tokens: int = 2048,
) -> list[str]:
    """Generate one response per problem using the current model state."""
    device = next(model.parameters()).device
    responses = []

    for problem in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{problem}\n\n"
                    "Please reason step by step, and put your final answer within \\boxed{}."
                ),
            },
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{problem}\n\nPlease reason step by step, "
                f"and put your final answer within \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_len:]
        responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return responses


def compute_epistemic_metrics(responses: list[str], tokenizer) -> dict:
    """Compute epistemic tracking metrics from a list of generated responses."""
    total_tokens = 0
    total_epistemic = 0

    for r in responses:
        tlen = len(tokenizer.encode(r))
        total_tokens += tlen
        total_epistemic += count_epistemic_tokens(r)

    return {
        "epistemic_per_1k_tokens": (total_epistemic / max(total_tokens, 1)) * 1000,
        "avg_response_length_tokens": total_tokens / max(len(responses), 1),
        "total_epistemic_count": total_epistemic,
        "total_output_tokens": total_tokens,
    }


def _print_eval_metrics(metrics: dict, label: str) -> None:
    print(f"  [{label}]")
    print(f"    Epistemic/1k tokens : {metrics['epistemic_per_1k_tokens']:.3f}")
    print(f"    Avg response length : {metrics['avg_response_length_tokens']:.1f} tokens")
    print(f"    Total epistemic hits: {metrics['total_epistemic_count']}")
    print(f"    Total output tokens : {metrics['total_output_tokens']}")


# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpistemicTokenCallback(TrainerCallback):
    """
    Tracks epistemic token metrics throughout training via generation.

    - on_train_begin : logs base model baseline (adapter disabled) + step-0 adapter reading
    - on_step_end    : logs every `eval_steps` steps
    - on_train_end   : logs final adapter metrics

    Key metrics logged:
      epistemic/adapter/per_1k_tokens    — the main signal; we want this increasing
      epistemic/adapter/avg_response_length
      epistemic/adapter/total_count
    """

    def __init__(
        self,
        tokenizer,
        eval_steps: int = 25,
        max_new_tokens: int = 512,
        use_wandb: bool = False,
        eval_problems: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.max_new_tokens = max_new_tokens
        self.use_wandb = use_wandb
        self.eval_problems = eval_problems or EVAL_PROBLEMS
        # Populated by on_step_end / on_train_end; read by run_sweep for summary
        self.last_metrics: dict = {}

    def _generate_and_log(self, model, label: str, step: int) -> dict:
        was_training = model.training
        model.eval()
        responses = generate_responses(
            model, self.tokenizer, self.eval_problems, self.max_new_tokens
        )
        metrics = compute_epistemic_metrics(responses, self.tokenizer)
        if was_training:
            model.train()

        print(f"\n[EpistemicTracker @ step {step}] ({label})")
        print(f"  Epistemic/1k tokens : {metrics['epistemic_per_1k_tokens']:.3f}")
        print(f"  Avg response length : {metrics['avg_response_length_tokens']:.1f} tokens")
        print(f"  Total epistemic hits: {metrics['total_epistemic_count']}")

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(
                        {
                            f"epistemic/{label}/per_1k_tokens": metrics["epistemic_per_1k_tokens"],
                            f"epistemic/{label}/avg_response_length": metrics["avg_response_length_tokens"],
                            f"epistemic/{label}/total_count": metrics["total_epistemic_count"],
                        },
                        step=step,
                    )
            except ImportError:
                pass

        return metrics

    def on_train_begin(
        self, _args, _state: TrainerState, _control: TrainerControl, model=None, **_kwargs
    ):
        print("\n[EpistemicTracker] Recording step-0 baseline …")

        # Base model with adapter disabled — reference point for the full run
        try:
            with model.disable_adapter():
                self._generate_and_log(model, "base_model", step=0)
        except Exception as e:
            print(f"[EpistemicTracker] Base model baseline failed: {e}")

        # Adapter at step 0 (weights still initialised to zero, should match base)
        self._generate_and_log(model, "adapter", step=0)

    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self.last_metrics = self._generate_and_log(
                model, "adapter", step=state.global_step
            )

    def on_train_end(
        self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs
    ):
        print(f"\n[EpistemicTracker] Final metrics at step {state.global_step}")
        self.last_metrics = self._generate_and_log(
            model, "adapter_final", step=state.global_step
        )


class AdapterVsBaseCallback(TrainerCallback):
    """
    At the end of each epoch, generates responses for a few eval problems with the
    LoRA adapter enabled and with it disabled, then prints a side-by-side comparison:
    response length, epistemic hit count, and the first 200 chars of each response.
    """

    def __init__(
        self,
        tokenizer,
        max_new_tokens: int = 512,
        num_problems: int = 3,
        eval_problems: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.eval_problems = (eval_problems or EVAL_PROBLEMS)[:num_problems]

    def on_epoch_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        print(f"\n{'='*72}")
        print(f"[AdapterVsBase] End of epoch {int(state.epoch)} — side-by-side comparison")
        print(f"{'='*72}")

        was_training = model.training
        model.eval()

        for i, problem in enumerate(self.eval_problems):
            short_q = problem[:80] + ("…" if len(problem) > 80 else "")
            print(f"\nProblem {i + 1}: {short_q}")

            # Adapter response
            adapter_resp = generate_responses(
                model, self.tokenizer, [problem], self.max_new_tokens
            )[0]
            a_len = len(self.tokenizer.encode(adapter_resp))
            a_ep = count_epistemic_tokens(adapter_resp)

            # Base model response (adapter weights suppressed)
            try:
                with model.disable_adapter():
                    base_resp = generate_responses(
                        model, self.tokenizer, [problem], self.max_new_tokens
                    )[0]
                b_len = len(self.tokenizer.encode(base_resp))
                b_ep = count_epistemic_tokens(base_resp)
            except Exception as exc:
                base_resp = f"[Error: {exc}]"
                b_len = b_ep = 0

            print(f"  {'Model':<10}  {'Tokens':>7}  {'Epistemic':>10}  Preview")
            print(f"  {'Adapter':<10}  {a_len:>7d}  {a_ep:>10d}  {adapter_resp[:200]!r}")
            print(f"  {'Base':<10}  {b_len:>7d}  {b_ep:>10d}  {base_resp[:200]!r}")

        if was_training:
            model.train()
        print(f"\n{'='*72}\n")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, return_metrics: bool = False) -> dict:
    """
    Run a single LoRA SFT training run.
    Returns a metrics dict when return_metrics=True (used by run_sweep).
    """
    print(f"\n{'='*60}")
    print(f"Base model : {args.base_model}")
    print(f"LoRA rank  : {args.lora_rank}  alpha: {args.lora_alpha}  dropout: {args.lora_dropout}")
    print(f"Output dir : {args.output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Format dataset ────────────────────────────────────────────────────────
    train_dataset = load_and_format_limo(tokenizer, args.max_seq_length)

    # Optional small validation split
    if args.val_split > 0:
        split = train_dataset.train_test_split(test_size=args.val_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)}  |  Val: {len(eval_dataset)}")
    else:
        eval_dataset = None
        print(f"Train: {len(train_dataset)}  |  No validation split")

    # ── Load model ────────────────────────────────────────────────────────────
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.load_in_4bit else None,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ── Wrap with LoRA ────────────────────────────────────────────────────────
    lora_cfg = build_lora_config(args)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        bf16=True,
        fp16=False,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        logging_steps=5,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"limo-lora-r{args.lora_rank}-a{args.lora_alpha}",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        packing=args.packing,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    epistemic_cb = EpistemicTokenCallback(
        tokenizer=tokenizer,
        eval_steps=args.epistemic_eval_steps,
        max_new_tokens=512,
        use_wandb=args.use_wandb,
    )
    adapter_vs_base_cb = AdapterVsBaseCallback(
        tokenizer=tokenizer,
        max_new_tokens=512,
        num_problems=3,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[epistemic_cb, adapter_vs_base_cb],
    )

    # Resume from checkpoint if available
    resume_from = None
    if args.resume_from_checkpoint:
        resume_from = args.resume_from_checkpoint
    elif os.path.isdir(args.output_dir):
        checkpoints = [
            d for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
        ]
        if checkpoints:
            resume_from = True  # auto-detect latest

    print("\nStarting training …")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to {args.output_dir}")

    # Save config for reproducibility
    cfg_path = os.path.join(args.output_dir, "lora_experiment_config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved → {cfg_path}")

    # ── Optional: merge & save standalone model ───────────────────────────────
    if args.merge_after_training:
        merge_and_save(args.output_dir, args.base_model, args.output_dir + "_merged")

    if return_metrics:
        # Extract final val loss from trainer log history
        val_loss = float("nan")
        for entry in reversed(trainer.state.log_history):
            if "eval_loss" in entry:
                val_loss = entry["eval_loss"]
                break
        return {
            "final_epistemic_per_1k": epistemic_cb.last_metrics.get(
                "epistemic_per_1k_tokens", float("nan")
            ),
            "final_val_loss": val_loss,
        }
    return {}


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(args):
    """
    Grid sweep over learning_rate × num_train_epochs.

    Grid:
      learning_rate   : [3e-5, 5e-5, 1e-4, 2e-4]
      num_train_epochs: [1, 2]

    Each combination writes to ./lora_sweep/lr{lr}_ep{epochs}/.
    If --use_wandb is set, each run is a separate wandb run within the same group.
    At the end, prints a summary table sorted by epistemic token ratio.
    """
    lr_grid = [3e-5, 5e-5, 1e-4, 2e-4]
    epoch_grid = [1, 2]
    group_name = f"limo-lora-sweep-r{args.lora_rank}"
    results = []

    for lr in lr_grid:
        for epochs in epoch_grid:
            run_args = copy.deepcopy(args)
            run_args.learning_rate = lr
            run_args.num_train_epochs = epochs
            run_args.output_dir = f"./lora_sweep/lr{lr:.0e}_ep{epochs}"
            run_args.sweep = False  # prevent accidental recursion

            print(f"\n{'#'*60}")
            print(f"# SWEEP: lr={lr:.0e}  epochs={epochs}")
            print(f"# Output: {run_args.output_dir}")
            print(f"{'#'*60}\n")

            if args.use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project="limo-lora",
                        group=group_name,
                        name=f"lr{lr:.0e}-ep{epochs}",
                        config=vars(run_args),
                        reinit=True,
                    )
                except Exception as e:
                    print(f"[Sweep] wandb init failed: {e}")

            metrics = train(run_args, return_metrics=True)
            results.append({
                "lr": lr,
                "epochs": epochs,
                "epistemic_per_1k": metrics.get("final_epistemic_per_1k", float("nan")),
                "val_loss": metrics.get("final_val_loss", float("nan")),
                "output_dir": run_args.output_dir,
            })

            if args.use_wandb:
                try:
                    import wandb
                    wandb.finish()
                except Exception:
                    pass

    # Summary table sorted by epistemic token ratio (descending)
    results.sort(key=lambda x: x["epistemic_per_1k"], reverse=True)
    print(f"\n{'='*72}")
    print("SWEEP RESULTS — sorted by epistemic token ratio (higher = more self-correction)")
    print(f"{'='*72}")
    print(f"{'LR':>10s}  {'Epochs':>6s}  {'Epist/1k':>10s}  {'Val Loss':>10s}  Output")
    print(f"{'-'*72}")
    for r in results:
        ep_str = f"{r['epistemic_per_1k']:.3f}" if r["epistemic_per_1k"] == r["epistemic_per_1k"] else "nan"
        vl_str = f"{r['val_loss']:.4f}" if r["val_loss"] == r["val_loss"] else "nan"
        print(f"{r['lr']:>10.0e}  {r['epochs']:>6d}  {ep_str:>10s}  {vl_str:>10s}  {r['output_dir']}")
    print(f"{'='*72}\n")


# ── Eval-only ─────────────────────────────────────────────────────────────────

def run_eval_only(args):
    """
    Load a saved LoRA adapter from --output_dir and run generation-based evaluation.

    Reports:
      - Epistemic token ratio (adapter vs base)
      - Average response length
      - Response length distribution
      - Delta between adapter and base model
    """
    from peft import PeftModel

    print(f"\n{'='*60}")
    print(f"Eval-only mode")
    print(f"Adapter : {args.output_dir}")
    print(f"Base    : {args.base_model}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.eval_problems_file:
        with open(args.eval_problems_file) as f:
            problems = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(problems)} problems from {args.eval_problems_file}")
    else:
        problems = EVAL_PROBLEMS
        print(f"Using {len(problems)} hardcoded eval problems")

    # Load base model once; we'll wrap it with the adapter afterwards so we
    # only need one model load for both evaluations.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = True
    base_model.eval()

    print("\n--- Base model (no adapter) ---")
    base_responses = generate_responses(base_model, tokenizer, problems)
    base_metrics = compute_epistemic_metrics(base_responses, tokenizer)
    _print_eval_metrics(base_metrics, "Base model")

    print("\n--- LoRA adapter model ---")
    adapter_model = PeftModel.from_pretrained(base_model, args.output_dir)
    adapter_model.eval()
    adapter_responses = generate_responses(adapter_model, tokenizer, problems)
    adapter_metrics = compute_epistemic_metrics(adapter_responses, tokenizer)
    _print_eval_metrics(adapter_metrics, "Adapter model")

    # Response length distribution
    lengths = sorted(len(tokenizer.encode(r)) for r in adapter_responses)
    n = len(lengths)
    print("\n--- Response length distribution (adapter) ---")
    print(f"  Min    : {lengths[0]:6d} tokens")
    print(f"  Median : {lengths[n // 2]:6d} tokens")
    print(f"  Max    : {lengths[-1]:6d} tokens")
    print(f"  Mean   : {sum(lengths) / n:6.1f} tokens")

    delta = adapter_metrics["epistemic_per_1k_tokens"] - base_metrics["epistemic_per_1k_tokens"]
    print(f"\nEpistemic ratio delta (adapter − base): {delta:+.3f} per 1k tokens")


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_and_save(lora_adapter_path: str, base_model_path: str, merged_output_path: str):
    """Merge LoRA adapter weights into the base model and save as a full HF model."""
    from peft import PeftModel

    print(f"\nMerging LoRA weights into base model …")
    print(f"  Adapter: {lora_adapter_path}")
    print(f"  Base   : {base_model_path}")
    print(f"  Output : {merged_output_path}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, lora_adapter_path)
    merged = model.merge_and_unload()

    os.makedirs(merged_output_path, exist_ok=True)
    merged.save_pretrained(merged_output_path)

    tok = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
    tok.save_pretrained(merged_output_path)

    print(f"Merged model saved → {merged_output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT on LIMO-v2 for Qwen3 math model")

    # Model
    parser.add_argument("--base_model", type=str,
                        default="beanie00/math-SDPO-Qwen3-8B-think-step-100")
    parser.add_argument("--output_dir", type=str, default="./lora_outputs/rank16")

    # LoRA hyperparameters (sweep these)
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank r. Try: 8, 16, 32, 64")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha. α/r=1 matches Schulman et al. calibration. Try: 16, 32, 64, 128")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout. Try: 0.0, 0.05, 0.1")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated list. Default: all attention+FFN projections")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Max sequence length for packing/truncation")
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Enable sequence packing (faster but masks cross-example boundaries)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of data to hold out for validation (0 = no val)")

    # Epistemic tracking
    parser.add_argument("--epistemic_eval_steps", type=int, default=25,
                        help="How often (in steps) to run epistemic token generation eval")

    # Hardware
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="QLoRA: load base model in 4-bit NF4")
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", dest="use_flash_attn", action="store_false")

    # Post-training
    parser.add_argument("--merge_after_training", action="store_true", default=False,
                        help="Merge LoRA into base weights and save full model")
    parser.add_argument("--merge_only", type=str, default=None,
                        help="Skip training; just merge adapter at this path into base model")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="Run grid sweep: lr ∈ [3e-5,5e-5,1e-4,2e-4] × epochs ∈ [1,2]")

    # Eval-only mode
    parser.add_argument("--eval_only", action="store_true", default=False,
                        help="Load adapter from --output_dir and run epistemic eval only (no training)")
    parser.add_argument("--eval_problems_file", type=str, default=None,
                        help="Path to a text file with one eval problem per line (used with --eval_only)")

    # Misc
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Stand-alone merge mode
    if args.merge_only:
        merge_and_save(
            lora_adapter_path=args.merge_only,
            base_model_path=args.base_model,
            merged_output_path=args.output_dir,
        )
        sys.exit(0)

    # Eval-only mode
    if args.eval_only:
        run_eval_only(args)
        sys.exit(0)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sweep:
        run_sweep(args)
    else:
        train(args)
