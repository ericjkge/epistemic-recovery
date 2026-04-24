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

### 0. System prerequisites (fresh server checklist)

Running on a fresh VM often trips on things outside Python. Check these **before** creating
the venv — in our setup each of these blocked `evaluate_aime.py` until fixed.

```bash
# Python 3.11 dev headers — Triton compiles a CUDA helper (driver.c) at runtime and needs Python.h.
# Without this, vLLM crashes with 'subprocess.check_call ... gcc ... returned non-zero exit status 1'.
sudo apt-get install -y python3.11-dev

# git-lfs — the LoRA adapter weights (saves/qwen3_sdpo_limo_lora/adapter_model.safetensors,
# ~175 MB) are stored via LFS. Without it, PEFT/vLLM see a pointer file and fail to load the adapter.
sudo apt-get install -y git-lfs
git lfs install
git lfs pull

# .config ownership — on some cloud images /home/$USER/.config is root-owned, which makes
# vLLM's usage-reporter crash with 'PermissionError: [Errno 13] ... /home/ubuntu/.config/vllm'.
sudo chown -R "$USER:$USER" "$HOME/.config"
```

Sanity-check:

```bash
ls /usr/include/python3.11/Python.h                                   # must exist
ls saves/qwen3_sdpo_limo_lora/adapter_model.safetensors               # must be ~175 MB (not a 100-byte pointer)
[ -w "$HOME/.config" ] && echo "config writable"
```

### 1. Create and activate

```bash
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
```

### 2. Install vLLM + evaluation deps

`requirements_eval.txt` pins `vllm==0.8.5.post1` for H100 / A100 / GH200 (CUDA 12.x). For
Blackwell (B100 / B200 / RTX 5090), edit the file to use `vllm>=0.12.0` instead.

> **Why 0.8.5.post1, not 0.8.4?** vLLM 0.8.4 ships a broken `LoRALRUCache` that crashes with
> `AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'` the moment a
> LoRA adapter is activated. 0.8.5.post1 fixes it.

```bash
pip install -r requirements_eval.txt
```

`requirements_eval.txt` also pins `transformers==4.51.3`. vLLM 0.8.5.post1's loose bound
would otherwise resolve to `transformers>=5.0`, which removed
`all_special_tokens_extended` and breaks vLLM's tokenizer init with
`AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`.

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

#### 3a. Merge the LoRA adapter (one-time, required on vLLM 0.8.5.post1)

vLLM 0.8.5.post1's Punica LoRA kernels segfault on H100 + CUDA 12.4 + driver 580 during
`profile_run` — the engine-core subprocess dies on SIGSEGV right after `Using PunicaWrapperGPU`,
with no Python traceback. Workaround: pre-merge the adapter into the base weights and load it as
a plain checkpoint (no Punica path involved at inference time).

```bash
source .venv-eval/bin/activate
python eval/merge_lora.py \
  --adapter "$PWD/saves/qwen3_sdpo_limo_lora" \
  --out     "$PWD/saves/qwen3_sdpo_limo_lora_merged"
```

Writes a ~16 GB merged checkpoint. Use absolute paths — PEFT treats relative paths as HF repo
IDs. `evaluate_aime.py` auto-detects `saves/qwen3_sdpo_limo_lora_merged/` and routes the `lora`
label to it; if the dir is absent, it falls back to base + adapter (which only works on setups
where Punica is healthy).

#### 3b. Run the evaluation

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

> **Note on `enforce_eager`:** the script passes `enforce_eager=True` to vLLM, which disables
> CUDA graphs / torch.compile. This sidesteps a class of compile-time failures we hit on H100
> + CUDA 12.4 at the cost of ~2–3× slower inference. If your server can run vLLM with graphs
> mode successfully, drop this kwarg in [eval/evaluate_aime.py](eval/evaluate_aime.py) for the speedup.

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

**`AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`** (vLLM startup)
: `transformers>=5.0` got installed. Pin it back: `pip install "transformers==4.51.3"`.
This is captured in `requirements_eval.txt`; re-sync if you upgraded anything recently.

**`AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'`**
: vLLM 0.8.4 bug in the LoRA cache. Upgrade: `pip install "vllm==0.8.5.post1"`, then re-pin
transformers as above (the upgrade may pull in a newer one).

**`Command ... gcc ... /tmp/.../main.c ... returned non-zero exit status 1`** during vLLM load
: Triton is trying to compile its CUDA driver helper and cannot find `Python.h`. Install
`python3.11-dev` (see the system prerequisites section). No venv rebuild needed — Triton
recompiles on demand.

**`Engine core initialization failed. See root cause above.`** with no visible error above
: The engine-core subprocess died on a signal (usually SIGSEGV), which produces no Python
traceback. Most commonly: vLLM's LoRA/Punica path crashing on H100 + CUDA 12.4 + driver 580.
Confirm by rerunning with `--models baseline` (no LoRA) — if baseline succeeds and `lora`
segfaults at `Using PunicaWrapperGPU`, follow Step 3a to merge the adapter and rerun.

**`PermissionError: [Errno 13] ... /home/ubuntu/.config/vllm`**
: `~/.config` is root-owned on some cloud images. Fix: `sudo chown -R "$USER:$USER" "$HOME/.config"`.
This is a background-thread warning and won't by itself stop the run, but it often appears alongside
the real failure and confuses the log.

**`adapter_model.safetensors` missing / `RepositoryNotFoundError` / PEFT treats path as HF repo ID**
: Two possible causes. (1) Git LFS wasn't set up — the adapter weights are a 175 MB LFS file;
run `sudo apt-get install -y git-lfs && git lfs install && git lfs pull`. (2) You passed a
relative path to PEFT; always use absolute paths (`"$PWD/saves/..."`) when calling
`eval/merge_lora.py` or `PeftModel.from_pretrained`.

**Training stalls / loss stays flat**
: The effective batch size is 8 (`per_device_train_batch_size=1` ×
`gradient_accumulation_steps=8`). If you have multiple GPUs, LlamaFactory will shard
automatically — no changes to the YAML needed.
