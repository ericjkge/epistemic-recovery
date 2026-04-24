"""Merge the LoRA adapter into the base model so vLLM can load it without Punica.

Workaround for a LoRA/Punica crash in vLLM 0.8.5.post1 on H100 + CUDA 12.4.

Run from sdpo_limo_llamafactory/ with the eval venv activated:
    source .venv-eval/bin/activate
    python eval/merge_lora.py
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "beanie00/math-SDPO-Qwen3-8B-think-step-100"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--adapter", default="saves/qwen3_sdpo_limo_lora")
    parser.add_argument("--out", default="saves/qwen3_sdpo_limo_lora_merged")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print(f"attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging...")
    model = model.merge_and_unload()
    print(f"saving to: {out}")
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("done.")


if __name__ == "__main__":
    main()
