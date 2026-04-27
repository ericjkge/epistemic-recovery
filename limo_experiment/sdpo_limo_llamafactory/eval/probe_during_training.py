#!/usr/bin/env python3
"""Epistemic alignment probe for LlamaFactory LoRA checkpoints.

Run this in a second terminal alongside `llamafactory-cli train` to evaluate
epistemic alignment at every saved checkpoint. It polls the adapter output
directory for new checkpoint-N subdirectories, attaches the LoRA adapter to the
base model, calls compute_epistemic_metrics, and logs scalars to wandb.

Before polling starts it runs two reference baselines:
  - base_qwen3_8b   : clean Qwen/Qwen3-8B (alignment target)
  - sdpo_no_lora    : the SDPO-suppressed checkpoint without any adapter (floor)

Reference scalars are logged as wandb constants so you can draw horizontal
reference lines in the dashboard and judge whether the LoRA is converging toward
the base-model alignment values.

The success criterion is matching base-model alignment, not over/under-shooting.

NOTE ON STEP GRANULARITY
  LlamaFactory's `save_steps` controls when checkpoint-N dirs are written.
  Set --eval_every_n_steps to match or be a multiple of save_steps in the YAML
  (default save_steps=200). Setting eval_every_n_steps=25 is only meaningful if
  you also lower save_steps to 25 in the training YAML (produces many checkpoints).

Usage:
    # In a separate terminal while training is running:
    python eval/probe_during_training.py \\
        --adapter_dir outputs/my_run \\
        --base_model beanie00/math-SDPO-Qwen3-8B-think-step-100 \\
        --probe_dataset limo_v2_sdpo.json \\
        --results_dir results/my_run/epistemic_probes \\
        --eval_every_n_steps 200 \\
        --wandb_project epistemic-recovery \\
        --wandb_run_name my_run_probes

    # Evaluate a single specific checkpoint and exit:
    python eval/probe_during_training.py \\
        --adapter_dir outputs/my_run \\
        --final_checkpoint outputs/my_run/checkpoint-1000 \\
        --results_dir results/my_run/epistemic_probes \\
        --skip_baselines
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from epistemic_metrics import (
    build_diagnostic_token_map,
    build_epistemic_token_ids,
    compute_epistemic_metrics,
    load_probe_set,
    prepare_probe_set,
)

PRETRAINED_MODEL = "Qwen/Qwen3-8B"


def _load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def _run_probe(
    model,
    tokenizer,
    prepared: list[dict],
    probe_traces: list[dict],
    ep_ids: torch.Tensor,
    diag_map: dict,
    label: str,
    step: int,
    results_dir: Path,
    wandb_run,
    reference_scalars: Optional[dict] = None,
) -> dict:
    t0 = time.time()
    print(f"\n  [probe] {label}  step={step}  n_traces={len(prepared)}", flush=True)

    metrics = compute_epistemic_metrics(
        model, tokenizer, probe_traces, ep_ids, diag_map, prepared=prepared,
    )
    elapsed = time.time() - t0

    scalars = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    arrays = {k: v for k, v in metrics.items() if isinstance(v, list)}

    print(
        f"  done in {elapsed:.1f}s — "
        f"gap={scalars['epistemic_alignment_gap']:.4f}  "
        f"entropy_ratio={scalars['epistemic_entropy_ratio']:.4f}  "
        f"trace_mean_lp={scalars['trace_mean_logprob']:.4f}",
        flush=True,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{label}_step{step:07d}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"label": label, "step": step, "elapsed_s": elapsed,
             "scalars": scalars, "arrays": arrays},
            f, indent=2,
        )
    print(f"  saved {out_path}", flush=True)

    if wandb_run is not None:
        log_dict: dict = {}
        for k, v in scalars.items():
            if isinstance(v, float) and v != v:
                continue
            log_dict[f"epistemic/{k}"] = v

        # Log reference baselines as constants alongside every training step
        # so they appear as horizontal lines in the wandb chart.
        if reference_scalars:
            for ref_label, ref_sc in reference_scalars.items():
                for k, v in ref_sc.items():
                    if isinstance(v, float) and v != v:
                        continue
                    log_dict[f"epistemic_ref/{ref_label}/{k}"] = v

        wandb_run.log(log_dict, step=step)

    return metrics


def _checkpoint_step(path: Path) -> Optional[int]:
    name = path.name
    prefix = "checkpoint-"
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return None


def main():
    parser = argparse.ArgumentParser(description="Epistemic probe for LlamaFactory LoRA checkpoints.")
    parser.add_argument("--base_model", default="beanie00/math-SDPO-Qwen3-8B-think-step-100")
    parser.add_argument("--pretrained_model", default=PRETRAINED_MODEL,
                        help="Clean pretrained model for baseline reference.")
    parser.add_argument("--adapter_dir", required=True,
                        help="Directory where LlamaFactory writes checkpoint-N subdirs.")
    parser.add_argument("--probe_dataset", default="limo_v2_sdpo.json")
    parser.add_argument("--n_probes", type=int, default=50)
    parser.add_argument("--results_dir", default="results/epistemic_probes")
    parser.add_argument("--eval_every_n_steps", type=int, default=200,
                        help="Only evaluate checkpoints at multiples of this value. "
                             "Should match or be a multiple of save_steps in the training YAML.")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between directory scans for new checkpoints.")
    parser.add_argument("--max_idle_minutes", type=int, default=0,
                        help="Stop polling after this many idle minutes (0 = run until Ctrl-C).")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip the pre-training baseline measurements.")
    parser.add_argument("--final_checkpoint", default=None,
                        help="Evaluate only this specific checkpoint directory, then exit.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    adapter_dir = Path(args.adapter_dir)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                resume="allow",
            )
            print(f"wandb: project={args.wandb_project}", flush=True)
        except ImportError:
            print("wandb not installed; skipping wandb logging.", flush=True)

    print(f"Loading probe set from {args.probe_dataset} (n={args.n_probes})...", flush=True)
    probe_traces = load_probe_set(args.probe_dataset, n_probes=args.n_probes, seed=42)

    # ── Pre-training reference baselines ──────────────────────────────────────
    reference_scalars: dict[str, dict] = {}

    if not args.skip_baselines:
        for bl_label, bl_path in [
            ("base_qwen3_8b", args.pretrained_model),
            ("sdpo_no_lora", args.base_model),
        ]:
            print(f"\nBaseline: {bl_label}  ({bl_path})", flush=True)
            tokenizer, model = _load_model(bl_path)
            ep_ids = build_epistemic_token_ids(tokenizer)
            diag_map = build_diagnostic_token_map(tokenizer)
            prepared = prepare_probe_set(tokenizer, probe_traces)

            bl_metrics = _run_probe(
                model, tokenizer, prepared, probe_traces, ep_ids, diag_map,
                label=bl_label, step=0,
                results_dir=results_dir, wandb_run=wandb_run,
            )
            reference_scalars[bl_label] = {
                k: v for k, v in bl_metrics.items() if not isinstance(v, list)
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()

        print("\nReference baselines computed:", flush=True)
        for bl_label, sc in reference_scalars.items():
            gap = sc.get("epistemic_alignment_gap", float("nan"))
            er = sc.get("epistemic_entropy_ratio", float("nan"))
            print(f"  {bl_label}: alignment_gap={gap:.4f}  entropy_ratio={er:.4f}", flush=True)
        print("Target: match base_qwen3_8b values (neither over- nor under-shooting).", flush=True)

    # ── Load base model for LoRA evaluation (kept resident across checkpoints) ─
    print(f"\nLoading SDPO base model for LoRA evaluation: {args.base_model}", flush=True)
    tokenizer, base_model = _load_model(args.base_model)
    ep_ids = build_epistemic_token_ids(tokenizer)
    diag_map = build_diagnostic_token_map(tokenizer)
    prepared = prepare_probe_set(tokenizer, probe_traces)

    # ── Single-checkpoint mode ────────────────────────────────────────────────
    if args.final_checkpoint:
        ckpt_path = Path(args.final_checkpoint)
        step = _checkpoint_step(ckpt_path) or 0
        from peft import PeftModel
        lora_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        _run_probe(
            lora_model, tokenizer, prepared, probe_traces, ep_ids, diag_map,
            label="lora", step=step,
            results_dir=results_dir, wandb_run=wandb_run,
            reference_scalars=reference_scalars,
        )
        if wandb_run:
            wandb_run.finish()
        return

    # ── Checkpoint polling loop ───────────────────────────────────────────────
    seen_steps: set[int] = set()
    last_new = time.time()

    print(
        f"\nMonitoring {adapter_dir}  "
        f"(eval_every={args.eval_every_n_steps} steps, poll={args.poll_interval}s)",
        flush=True,
    )

    try:
        while True:
            if adapter_dir.exists():
                pending = []
                for child in sorted(adapter_dir.iterdir()):
                    step = _checkpoint_step(child)
                    if step is None or step in seen_steps:
                        continue
                    seen_steps.add(step)
                    if step % args.eval_every_n_steps == 0:
                        pending.append((step, child))

                for step, ckpt_path in pending:
                    last_new = time.time()
                    print(f"\nNew checkpoint: {ckpt_path}  (step {step})", flush=True)

                    from peft import PeftModel
                    lora_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
                    _run_probe(
                        lora_model, tokenizer, prepared, probe_traces, ep_ids, diag_map,
                        label="lora", step=step,
                        results_dir=results_dir, wandb_run=wandb_run,
                        reference_scalars=reference_scalars,
                    )
                    del lora_model
                    gc.collect()
                    torch.cuda.empty_cache()

            if args.max_idle_minutes > 0:
                idle_min = (time.time() - last_new) / 60.0
                if idle_min > args.max_idle_minutes:
                    print(f"No new checkpoints for {idle_min:.0f} min — stopping.", flush=True)
                    break

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
