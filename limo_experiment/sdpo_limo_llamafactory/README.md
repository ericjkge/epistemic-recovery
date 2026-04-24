# SDPO + LIMO LoRA Fine-Tuning

This directory contains the full pipeline for fine-tuning an SDPO-trained Qwen3-8B with LIMO
to recover epistemic verbalization, and evaluating recovery on AIME 2024/2025.

---

## Why Two Environments

Training (LlamaFactory) and evaluation (vLLM) have incompatible transitive dependencies on
`transformers` and `torch`. Installing them together causes import errors at runtime. Use two
separate virtual environments and keep them strictly separate.

| venv | Directory | Used for |
|---|---|---|
| Training | `.venv-train/` | `prepare_limo.py`, LlamaFactory SFT, `eval/analyze_epistemic.py`, `eval/analyze_token_distribution.py` |
| Evaluation | `.venv-eval/` | `eval/evaluate_aime.py` (vLLM backend) |

Both directories are gitignored. All commands below assume you are in `sdpo_limo_llamafactory/`.

---

## Environment 1: Training (`.venv-train`)

### 1. Create and activate

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
```

### 2. Install PyTorch with CUDA

Match the CUDA version on your machine. For CUDA 12.x (H100 / GH200):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check your CUDA version with `nvidia-smi`. Adjust `cu121` → `cu118` etc. as needed.

### 3. Clone and install LlamaFactory

LlamaFactory is not vendored in this repo — clone it into the `LlamaFactory/` directory:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git LlamaFactory/
cd LlamaFactory
pip install -e ".[torch,metrics]"
cd ..
```

> `pip install -e ".[torch,metrics]"` pulls in `transformers`, `accelerate`, `peft`,
> `datasets`, and `tqdm`. Run it before the next step so it doesn't downgrade anything.

### 4. Install flash-attn

```bash
pip install flash-attn --no-build-isolation
```

This can take 10–20 minutes to compile. If you hit a CUDA mismatch error, make sure the
PyTorch CUDA version from step 2 matches the system CUDA toolkit (`nvidia-smi`).

### 5. Install remaining requirements

```bash
pip install -r requirements.txt
```

### 6. Verify

```bash
python -c "import transformers, peft, datasets; print('OK')"
llamafactory-cli version
```

---

## Environment 2: Evaluation (`.venv-eval`)

vLLM ships with its own pinned versions of `torch` and `transformers`. Installing it into a
clean environment avoids all conflicts.

### 1. Create and activate

```bash
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
```

### 2. Install vLLM + evaluation deps

Open `requirements_eval.txt` and uncomment the correct vLLM line for your hardware first:
- `vllm==0.8.4` — H100 / A100 / GH200
- `vllm>=0.12.0` — Blackwell (B100 / B200 / RTX 5090)

Then install:

```bash
pip install -r requirements_eval.txt
```

vLLM will pull in compatible versions of `torch` and `transformers` automatically.

### 3. Verify

```bash
python -c "from vllm import LLM; print('vLLM OK')"
python -c "from peft import PeftModel; print('peft OK')"
```

---

## Pipeline

### Step 1 — Prepare LIMO data (training env)

Converts [GAIR/LIMO](https://huggingface.co/datasets/GAIR/LIMO) into Qwen3 thinking-mode
format by wrapping the reasoning trace in `<think>...</think>`.

```bash
source .venv-train/bin/activate
python prepare_limo.py \
  --dataset GAIR/LIMO \
  --output data/limo_qwen3_thinking.json \
  --tokenizer_name Qwen/Qwen3-8B
```

`data/limo_qwen3_thinking.json` is already committed to the repo — skip this step if it exists.

### Step 2 — LoRA SFT via LlamaFactory (training env)

Before running, update `dataset_dir` in `qwen3_sdpo_lora_sft.yaml` to the absolute path of
this directory on your machine:

```yaml
dataset_dir: /absolute/path/to/sdpo_limo_llamafactory
```

Then launch training:

```bash
source .venv-train/bin/activate
llamafactory-cli train qwen3_sdpo_lora_sft.yaml
```

For multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train qwen3_sdpo_lora_sft.yaml
```

Checkpoints are saved to `outputs/sdpo-qwen3-8b-limo-lora-r16/`. Copy the final adapter to
the path the evaluation script expects:

```bash
cp -r outputs/sdpo-qwen3-8b-limo-lora-r16 saves/qwen3_sdpo_limo_lora
```

Key training hyperparameters (see `qwen3_sdpo_lora_sft.yaml` for the full config):

| Parameter | Value |
|---|---|
| Base model | `beanie00/math-SDPO-Qwen3-8B-think-step-100` |
| LoRA rank | 16 |
| LoRA targets | all attention + FFN projections |
| Sequence length | 16384 tokens |
| Epochs | 2 |
| Learning rate | 5e-5 |
| Batch size (effective) | 8 (1 per device × 8 grad accumulation steps) |

### Step 3 — Evaluate on AIME (evaluation env)

Evaluates three model variants on AIME 2024 and AIME 2025 with n=16 completions per problem.

```bash
source .venv-eval/bin/activate
python eval/evaluate_aime.py \
  --adapter_path saves/qwen3_sdpo_limo_lora \
  --results_dir results \
  --n_sampling 16 \
  --tensor_parallel_size 1    # set to number of GPUs available
```

To evaluate only a subset:

```bash
# Only run the LoRA model on AIME 2024
python eval/evaluate_aime.py --models lora --benchmarks aime24

# Quick smoke test (2 problems, 2 samples)
python eval/evaluate_aime.py --max_questions 2 --n_sampling 2
```

Results are written to `results/{label}_{benchmark}.json`.

### Step 4 — Epistemic analysis (training env)

Counts the Kim et al. 10-token set (`wait`, `hmm`, `perhaps`, `maybe`, `actually`,
`alternatively`, `seems`, `might`, `likely`, `check`) inside each generation's
`<think>` span. Produces a CSV summary, per-token bar chart, and length-vs-accuracy scatter.

```bash
source .venv-train/bin/activate
python eval/analyze_epistemic.py --results_dir results
```

Outputs:
- `results/epistemic_summary.csv` — per-model × per-benchmark stats
- `results/epistemic_comparison.png` — bar chart of epistemic token counts
- `results/length_vs_accuracy.png` — scatter plot

### Step 5 — Token distribution analysis (training env)

Computes **teacher-forced token-level log probabilities and Shannon entropy** over a dataset.
This is the deepest signal for whether the LoRA has recovered epistemic verbalization: higher
log-prob and lower entropy at epistemic token positions post-LoRA means the model assigns more
decisive probability mass to these verbalizations.

All 10 epistemic tokens (both lowercase and capitalized variants) are tracked by default.

**Option A — analyze gold references from the training set (measures internalization):**

```bash
source .venv-train/bin/activate

# Pre-LoRA baseline
python eval/analyze_token_distribution.py \
  --model_name_or_path beanie00/math-SDPO-Qwen3-8B-think-step-100 \
  --input_json data/limo_qwen3_thinking.json \
  --tag pre_lora

# Post-LoRA (merged checkpoint)
python eval/analyze_token_distribution.py \
  --model_name_or_path saves/qwen3_sdpo_limo_lora/merged \
  --input_json data/limo_qwen3_thinking.json \
  --tag post_lora

# Post-LoRA (base + adapter, no merge needed)
python eval/analyze_token_distribution.py \
  --model_name_or_path beanie00/math-SDPO-Qwen3-8B-think-step-100 \
  --adapter_path saves/qwen3_sdpo_limo_lora \
  --input_json data/limo_qwen3_thinking.json \
  --tag post_lora
```

**Option B — analyze the model's own generated AIME outputs (measures actual behavior):**

```bash
python eval/analyze_token_distribution.py \
  --model_name_or_path beanie00/math-SDPO-Qwen3-8B-think-step-100 \
  --adapter_path saves/qwen3_sdpo_limo_lora \
  --results_json results/lora_sdpo_plus_limo_aime25.json \
  --tag post_lora_aime25_generated
```

All outputs are written to `results/token_dist/{tag}/`:

| File | Contents |
|---|---|
| `{tag}_token_stats.csv` | Per-token-type aggregates: count, avg logprob, avg entropy |
| `{tag}_special_tokens.csv` | Per-occurrence logprob/entropy for each epistemic token |
| `{tag}_all_logprobs.csv` | Full logprob histogram (overlay pre/post for distribution shift) |
| `{tag}_all_entropies.csv` | Full entropy histogram |
| `{tag}_logprob_dist.png` | Histogram + CDF + box plot + summary stats |
| `{tag}_special_token_dist.png` | Logprob/entropy density curves: all tokens vs. each epistemic token |

**What to look for when comparing pre vs. post LoRA:**
- `avg_logprob` for epistemic tokens should be **higher** (less negative) post-LoRA
- `avg_entropy` at epistemic token positions should be **lower** post-LoRA
- The overall logprob distribution (`_all_logprobs.csv`) should shift right if the model has become more confident on the training distribution

---

## Troubleshooting

**`ImportError: cannot import name X from transformers`** during evaluation
: You are likely in `.venv-train` instead of `.venv-eval`. Run
`source .venv-eval/bin/activate` and retry.

**`llamafactory-cli: command not found`**
: LlamaFactory is not installed in the active environment. Make sure you ran
`pip install -e ".[torch,metrics]"` from inside `LlamaFactory/` while `.venv-train` was active.

**`flash_attn` compile error**
: PyTorch CUDA version must match the system CUDA toolkit. Check with
`python -c "import torch; print(torch.version.cuda)"` and `nvidia-smi`. Reinstall PyTorch
with the matching `--index-url` if they differ.

**vLLM OOM on a single GPU**
: Reduce `--gpu_memory_utilization` (default 0.90) or decrease `--max_model_len` (default
40960). For multi-GPU, increase `--tensor_parallel_size`.

**Training stalls / loss stays flat**
: The effective batch size is 8 (`per_device_train_batch_size=1` ×
`gradient_accumulation_steps=8`). If you have multiple GPUs, LlamaFactory will shard
automatically — no changes to the YAML needed.
