#!/usr/bin/env bash
# =============================================================================
# Full LIMO LoRA Pipeline
# =============================================================================
# Steps:
#   1. Download model  (beanie00/math-SDPO-Qwen3-8B-think-step-100)
#   2. Download LIMO-v2 dataset (cached by HuggingFace automatically)
#   3. Pre-LoRA AIME epistemic evaluation
#   4. LoRA SFT on LIMO-v2  (run multiple ranks as a sweep)
#   5. Merge LoRA weights → full model for each run
#   6. Post-LoRA AIME epistemic evaluation
#   7. Compare pre vs. post results
#
# Requirements: run from limo_experiment/ with the conda/venv already activated.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurable paths ────────────────────────────────────────────────────────
BASE_MODEL="beanie00/math-SDPO-Qwen3-8B-think-step-100"
LOCAL_MODEL_DIR="./models/math-SDPO-Qwen3-8B-think-step-100"
RESULTS_DIR="./results"
LORA_BASE_DIR="./lora_outputs"

# ── GPU config (edit as needed) ───────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
echo "Using $N_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"

# =============================================================================
# STEP 1 — Download the model
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 1: Downloading model → $LOCAL_MODEL_DIR"
echo "============================================================"

if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    python - <<'EOF'
import sys, os
from huggingface_hub import snapshot_download
model_id = "beanie00/math-SDPO-Qwen3-8B-think-step-100"
local_dir = "./models/math-SDPO-Qwen3-8B-think-step-100"
print(f"Downloading {model_id} …")
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
print(f"Model saved to {local_dir}")
EOF
else
    echo "  Model already present at $LOCAL_MODEL_DIR — skipping download."
fi

# =============================================================================
# STEP 2 — Download / cache LIMO-v2
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 2: Pre-caching LIMO-v2 dataset"
echo "============================================================"

python - <<'EOF'
from datasets import load_dataset
print("Downloading GAIR/LIMO-v2 …")
ds = load_dataset("GAIR/LIMO-v2", split="train")
print(f"Dataset columns : {ds.column_names}")
print(f"Dataset size    : {len(ds)} examples")
EOF

# =============================================================================
# STEP 3 — Pre-LoRA AIME Epistemic Evaluation
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 3: Pre-LoRA epistemic evaluation on AIME 2024 + 2025"
echo "============================================================"

PRE_RESULTS_DIR="$RESULTS_DIR/pre_lora"
mkdir -p "$PRE_RESULTS_DIR"

python aime_epistemic_eval.py \
    --model_path "$LOCAL_MODEL_DIR" \
    --aime_years 2024 2025 \
    --output_dir "$PRE_RESULTS_DIR" \
    --n_samples 4 \
    --temperature 0.6 \
    --max_tokens 32768 \
    --enable_thinking

echo "Pre-LoRA results saved to $PRE_RESULTS_DIR"

# =============================================================================
# STEP 4 — LoRA SFT Sweep
# =============================================================================
# We train three variants to compare rank effects:
#   - rank  8, alpha 16   (lightweight)
#   - rank 32, alpha 64   (default)
#   - rank 64, alpha 128  (high-capacity)
#
# Uncomment the variants you want to run. H100/A100 can handle all three
# sequentially; on smaller GPUs use only one at a time.
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 4: LoRA SFT on LIMO-v2"
echo "============================================================"

run_lora() {
    local RANK=$1
    local ALPHA=$2
    local EPOCHS=$3
    local RUN_NAME="rank${RANK}_alpha${ALPHA}_ep${EPOCHS}"
    local OUT_DIR="$LORA_BASE_DIR/$RUN_NAME"

    echo ""
    echo "--- Training: $RUN_NAME ---"
    python limo_lora_sft.py \
        --base_model         "$LOCAL_MODEL_DIR" \
        --output_dir         "$OUT_DIR" \
        --lora_rank          "$RANK" \
        --lora_alpha         "$ALPHA" \
        --lora_dropout       0.05 \
        --num_train_epochs   "$EPOCHS" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate      2e-4 \
        --max_seq_length     8192 \
        --warmup_ratio       0.05 \
        --val_split          0.05 \
        --use_flash_attn \
        --merge_after_training   # produces ${OUT_DIR}_merged/

    echo "LoRA adapter → $OUT_DIR"
    echo "Merged model → ${OUT_DIR}_merged"
}

# Run the default rank-32 config
run_lora 32 64 3

# Uncomment for additional sweep points:
# run_lora 8  16 3
# run_lora 64 128 3

# =============================================================================
# STEP 5 — Post-LoRA AIME Epistemic Evaluation
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 5: Post-LoRA epistemic evaluation on AIME 2024 + 2025"
echo "============================================================"

# Evaluate each merged model found in lora_outputs/
for MERGED_DIR in "$LORA_BASE_DIR"/*_merged; do
    if [ ! -d "$MERGED_DIR" ]; then
        continue
    fi
    RUN_TAG=$(basename "$MERGED_DIR")
    POST_RESULTS_DIR="$RESULTS_DIR/post_lora/$RUN_TAG"
    mkdir -p "$POST_RESULTS_DIR"

    echo ""
    echo "Evaluating: $MERGED_DIR"
    python aime_epistemic_eval.py \
        --model_path    "$MERGED_DIR" \
        --aime_years    2024 2025 \
        --output_dir    "$POST_RESULTS_DIR" \
        --n_samples     4 \
        --temperature   0.6 \
        --max_tokens    32768 \
        --enable_thinking
done

# =============================================================================
# STEP 6 — Compare Pre vs. Post Results
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 6: Comparing pre vs. post LoRA results"
echo "============================================================"

python compare_results.py \
    --pre_dir  "$PRE_RESULTS_DIR" \
    --post_dir "$RESULTS_DIR/post_lora"

echo ""
echo "Pipeline complete.  All results in: $RESULTS_DIR"
