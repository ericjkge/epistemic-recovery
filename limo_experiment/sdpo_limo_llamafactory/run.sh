#!/usr/bin/env bash
# Full pipeline: LIMO-v2 data prep → LoRA SFT → merge → AIME eval → epistemic analysis.
#
# Each run writes to results/{RUN_ID}/ and outputs/{RUN_NAME}/ so repeated runs
# never clobber each other. The 2026-04-24 results live in results/ and
# outputs/sdpo-qwen3-8b-limo-lora-r16 and are preserved as-is.
#
# Usage:
#   ./run.sh                                         # all defaults
#   N_SAMPLING=16 MODELS=baseline,lora,pretrained ./run.sh
#   RUN_ID=ablate_1epoch N_EPOCHS=1 ./run.sh
#   SKIP_TRAIN=1 RUN_ID=20260424_000000 ./run.sh    # re-run eval on existing adapter

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ── Identifiers ───────────────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-sdpo_limo_lora_${RUN_ID}}"

# ── Training hyperparameters ──────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-beanie00/math-SDPO-Qwen3-8B-think-step-100}"
N_EPOCHS="${N_EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-5e-5}"
CUTOFF_LEN="${CUTOFF_LEN:-16384}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
BASE_YAML="qwen3_sdpo_lora_sft.yaml"

# ── Path layout ───────────────────────────────────────────────────────────────
ADAPTER_DIR="${ADAPTER_DIR:-outputs/${RUN_NAME}}"
MERGED_DIR="${MERGED_DIR:-${ADAPTER_DIR}/merged}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_ID}}"

# ── Eval / sampling ──────────────────────────────────────────────────────────
# Recs from EXPERIMENT_NOTES: bump to 16 for tighter bounds; 4 is fast smoke-test.
N_SAMPLING="${N_SAMPLING:-4}"
MAX_TOKENS="${MAX_TOKENS:-24576}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.3}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25}"
# Add "pretrained" to include Qwen3-8B upper bound (recommended but expensive).
MODELS="${MODELS:-baseline,lora}"

# ── Epistemic alignment probing ───────────────────────────────────────────────
# Generates Figure-9-style diagnostic plots after training:
#   CDF of per-trace mean log-prob + log-prob/entropy density grids per model.
# During training, run eval/probe_during_training.py in a separate terminal
# (see that script's usage for per-checkpoint wandb logging).
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen3-8B}"
N_PROBES="${N_PROBES:-50}"
WANDB_PROJECT="${WANDB_PROJECT:-}"      # set to enable wandb logging in probe scripts

# ── Loop detection ────────────────────────────────────────────────────────────
# Terminates degenerate repetition sequences early (saves compute on the ~25% of
# LoRA responses that loop to max_tokens). Set LOOP_WINDOW=0 to disable.
LOOP_WINDOW="${LOOP_WINDOW:-300}"
LOOP_THRESHOLD="${LOOP_THRESHOLD:-0.35}"
LOOP_MIN_TOKENS="${LOOP_MIN_TOKENS:-100}"

# ── Skip flags (set to 1 to restart from a later step) ───────────────────────
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EPISTEMIC="${SKIP_EPISTEMIC:-0}"

PY="${PYTHON:-python3}"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

cat > "$RESULTS_DIR/run_config.txt" <<EOF
RUN_ID             $RUN_ID
RUN_NAME           $RUN_NAME
BASE_MODEL         $BASE_MODEL
ADAPTER_DIR        $ADAPTER_DIR
MERGED_DIR         $MERGED_DIR
RESULTS_DIR        $RESULTS_DIR
N_EPOCHS           $N_EPOCHS
LORA_RANK          $LORA_RANK
LR                 $LR
CUTOFF_LEN         $CUTOFF_LEN
GRAD_ACCUM         $GRAD_ACCUM
N_SAMPLING         $N_SAMPLING
MAX_TOKENS         $MAX_TOKENS
FREQUENCY_PENALTY  $FREQUENCY_PENALTY
BENCHMARKS         $BENCHMARKS
MODELS             $MODELS
PRETRAINED_MODEL   $PRETRAINED_MODEL
N_PROBES           $N_PROBES
WANDB_PROJECT      $WANDB_PROJECT
DATE               $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo "Config → $RESULTS_DIR/run_config.txt"

# ── Step 1: Dataset prep ──────────────────────────────────────────────────────
echo ""
echo "[1/5] LIMO-v2 dataset prep"
if [ "$SKIP_DATA" = "1" ]; then
    echo "  SKIP_DATA=1 — reusing existing limo_v2_sdpo.json"
else
    $PY make_limo_dataset.py --output limo_v2_sdpo.json
    echo "  wrote limo_v2_sdpo.json"
fi

# ── Step 2: LoRA SFT ──────────────────────────────────────────────────────────
echo ""
echo "[2/5] LoRA SFT (LlamaFactory)"
if [ "$SKIP_TRAIN" = "1" ]; then
    echo "  SKIP_TRAIN=1 — using existing adapter at $ADAPTER_DIR"
    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "ERROR: ADAPTER_DIR not found: $ADAPTER_DIR"
        exit 1
    fi
else
    # Patch the YAML into a temp file so the source template is never modified.
    PATCHED_YAML="$(mktemp /tmp/sdpo_limo_XXXXXX.yaml)"
    sed \
        -e "s|dataset_dir:.*|dataset_dir: ${HERE}|" \
        -e "s|output_dir:.*|output_dir: ${ADAPTER_DIR}|" \
        -e "s|num_train_epochs:.*|num_train_epochs: ${N_EPOCHS}|" \
        -e "s|lora_rank:.*|lora_rank: ${LORA_RANK}|" \
        -e "s|learning_rate:.*|learning_rate: ${LR}|" \
        -e "s|cutoff_len:.*|cutoff_len: ${CUTOFF_LEN}|" \
        -e "s|gradient_accumulation_steps:.*|gradient_accumulation_steps: ${GRAD_ACCUM}|" \
        "$BASE_YAML" > "$PATCHED_YAML"
    cp "$PATCHED_YAML" "$RESULTS_DIR/training_config.yaml"
    echo "  effective config → $RESULTS_DIR/training_config.yaml"
    llamafactory-cli train "$PATCHED_YAML"
    rm -f "$PATCHED_YAML"
    echo "  adapter → $ADAPTER_DIR"
fi

# ── Step 3: Merge LoRA ────────────────────────────────────────────────────────
# Workaround for vLLM/Punica segfault on H100 + CUDA 12.4 (see EXPERIMENT_NOTES.md).
# If vLLM LoRA works on the target server, set SKIP_MERGE=1 and pass --models with
# the raw adapter; evaluate_aime.py will load base+adapter directly.
echo ""
echo "[3/5] Merging LoRA adapter"
if [ "$SKIP_MERGE" = "1" ]; then
    echo "  SKIP_MERGE=1 — skipping merge; eval will load base+adapter via vLLM LoRA"
    MERGED_DIR=""
else
    $PY eval/merge_lora.py \
        --adapter "$ADAPTER_DIR" \
        --out "$MERGED_DIR"
    echo "  merged model → $MERGED_DIR"
fi

# ── Step 4: AIME evaluation ───────────────────────────────────────────────────
echo ""
echo "[4/5] AIME evaluation (n=${N_SAMPLING}, benchmarks=${BENCHMARKS})"
if [ "$SKIP_EVAL" = "1" ]; then
    echo "  SKIP_EVAL=1 — skipping evaluation"
else
    EVAL_ARGS=(
        --adapter_path "$ADAPTER_DIR"
        --results_dir "$RESULTS_DIR"
        --benchmarks "$BENCHMARKS"
        --models "$MODELS"
        --n_sampling "$N_SAMPLING"
        --max_tokens "$MAX_TOKENS"
        --frequency_penalty "$FREQUENCY_PENALTY"
        --loop_window "$LOOP_WINDOW"
        --loop_threshold "$LOOP_THRESHOLD"
        --loop_min_tokens "$LOOP_MIN_TOKENS"
    )
    if [ -n "${MERGED_DIR:-}" ] && [ -d "${MERGED_DIR:-}" ]; then
        EVAL_ARGS+=(--lora_merged_path "$MERGED_DIR")
    fi
    $PY eval/evaluate_aime.py "${EVAL_ARGS[@]}"
fi

# ── Step 5: Epistemic analysis + figures ─────────────────────────────────────
echo ""
echo "[5/5] Epistemic analysis + figures"
$PY eval/analyze_epistemic.py \
    --results_dir "$RESULTS_DIR" \
    --csv_out "$RESULTS_DIR/epistemic_summary.csv" \
    --bars_out "$RESULTS_DIR/epistemic_comparison.png" \
    --scatter_out "$RESULTS_DIR/length_vs_accuracy.png" \
    --epistemic_bars_out "$RESULTS_DIR/epistemic_per_response.png" \
    --accuracy_bars_out "$RESULTS_DIR/accuracy_comparison.png"

# ── Step 6: Epistemic alignment probing + Figure 9 ────────────────────────────
# Computes token log-prob / entropy metrics for three models and generates
# Figure-9-style diagnostic plots (CDF + density grid).
#
# Two paths:
#   a) Merged model exists → call evaluate_epistemic_alignment.py in --models mode
#      directly (it treats the merged dir as a plain HF model, no PEFT needed).
#   b) No merged model → use probe_during_training.py on the final checkpoint dir
#      (loads base + LoRA adapter via PeftModel).
#
# Either way, previously saved probe JSON snapshots (from running
# probe_during_training.py during training) are preferred to avoid recomputing.
#
# NOTE: This step loads models sequentially and is slow (~10 min/model on H100).
# For per-step wandb logging during training, run probe_during_training.py in a
# second terminal alongside llamafactory-cli train.
echo ""
echo "[6/6] Epistemic alignment evaluation (Figure 9)"
if [ "$SKIP_EPISTEMIC" = "1" ]; then
    echo "  SKIP_EPISTEMIC=1 — skipping epistemic alignment evaluation"
else
    EPISTEMIC_DIR="$RESULTS_DIR/epistemic_probes"
    FIGURE9_OUT="$RESULTS_DIR/figure9.pdf"

    # If probe_during_training.py was already run during training, its JSON
    # snapshots are in EPISTEMIC_DIR. Use them directly to skip model loading.
    PROBE_JSONS=("${EPISTEMIC_DIR}"/*.json)
    if [ ${#PROBE_JSONS[@]} -gt 0 ] && [ -f "${PROBE_JSONS[0]}" ]; then
        echo "  Found ${#PROBE_JSONS[@]} pre-computed probe snapshot(s) — generating Figure 9 directly."
        $PY eval/evaluate_epistemic_alignment.py \
            --from_json "${PROBE_JSONS[@]}" \
            --output "$FIGURE9_OUT"
        echo "  figure9 → $FIGURE9_OUT"
    else
        # No pre-computed snapshots; compute metrics fresh.
        # Determine the LoRA model spec for evaluate_epistemic_alignment.py.
        if [ -n "${MERGED_DIR:-}" ] && [ -d "${MERGED_DIR:-}" ]; then
            # Merged checkpoint: plain HF model, no PEFT wrapper needed.
            LORA_SPEC="lora:${MERGED_DIR}"
        else
            # Unmerged: base + adapter (loaded via PeftModel inside the script).
            FINAL_CKPT="$(ls -d "${ADAPTER_DIR}"/checkpoint-* 2>/dev/null \
                          | sort -t- -k2 -n | tail -1)"
            if [ -n "$FINAL_CKPT" ]; then
                LORA_SPEC="lora:${BASE_MODEL}+${FINAL_CKPT}"
            else
                LORA_SPEC="lora:${BASE_MODEL}+${ADAPTER_DIR}"
            fi
        fi

        echo "  Computing metrics fresh (slow — ~10 min/model on H100)..."
        echo "    pretrained : $PRETRAINED_MODEL"
        echo "    sdpo       : $BASE_MODEL"
        echo "    lora       : ${LORA_SPEC#lora:}"

        $PY eval/evaluate_epistemic_alignment.py \
            --probe_dataset limo_v2_sdpo.json \
            --n_probes "$N_PROBES" \
            --models \
                "pretrained:${PRETRAINED_MODEL}" \
                "sdpo:${BASE_MODEL}" \
                "${LORA_SPEC}" \
            --output "$FIGURE9_OUT"
        echo "  figure9 → $FIGURE9_OUT"
    fi
fi

echo ""
echo "Done. All outputs in $RESULTS_DIR/"
echo ""
ls -1 "$RESULTS_DIR/"
