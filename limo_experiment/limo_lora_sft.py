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
      --output_dir ./lora_outputs/rank32 \
      --lora_rank 32 \
      --lora_alpha 64 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --learning_rate 2e-4

After training, use --merge to fuse LoRA weights into a standalone HF model.
"""

import argparse
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
      lora_alpha    : typically 2×rank or equal to rank
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
        # Merge into base weights after training for clean checkpointing
        modules_to_save=None,
    )


def train(args):
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
        logging_steps=10,
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

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
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
    parser.add_argument("--output_dir", type=str, default="./lora_outputs/rank32")

    # LoRA hyperparameters (sweep these)
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank r. Try: 8, 16, 32, 64")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha. Typically 2×rank. Try: 16, 32, 64, 128")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout. Try: 0.0, 0.05, 0.1")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated list. Default: all attention+FFN projections")

    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Max sequence length for packing/truncation")
    parser.add_argument("--packing", action="store_true", default=False,
                        help="Enable sequence packing (faster but masks cross-example boundaries)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of data to hold out for validation (0 = no val)")

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

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)
