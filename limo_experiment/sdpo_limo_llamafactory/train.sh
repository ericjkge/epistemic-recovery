#!/usr/bin/env bash
# Data prep → LoRA SFT → merge LoRA, with epistemic probe monitor running alongside.
#
# Writes the EXPERIMENT name to .last_experiment so eval.sh can pick it up
# without you having to pass it manually.
#
# Usage:
#   ./train.sh                                      # auto-pick next experimentN
#   EXPERIMENT=experiment3 ./train.sh               # explicit experiment name
#   N_EPOCHS=1 ./train.sh                           # override hparam
#   SKIP_DATA=1 SKIP_TRAIN=1 ./train.sh             # re-merge an existing adapter
#
# Epistemic probe monitor:
#   If you're inside a tmux session, train.sh automatically splits a new pane and
#   starts probe_during_training.py there before training begins. The probe computes
#   reference baselines (base Qwen3-8B + SDPO) then polls for new checkpoints.
#   Outside tmux it prints the command to run in a second terminal.
#
#   Set WANDB_PROJECT to enable wandb logging for the probe curves.
#   Set SKIP_PROBE=1 to disable the probe entirely.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ── Experiment identifier ─────────────────────────────────────────────────────
# Auto-pick the next free experiments/experimentN/ unless EXPERIMENT is set.
next_experiment_num() {
    local n=1
    while [ -d "experiments/experiment${n}" ]; do
        n=$((n+1))
    done
    echo "experiment${n}"
}
EXPERIMENT="${EXPERIMENT:-$(next_experiment_num)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# ── Model paths ───────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-beanie00/math-SDPO-Qwen3-8B-think-step-100}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen3-8B}"

# ── Training hyperparameters ──────────────────────────────────────────────────
N_EPOCHS="${N_EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-16}"
LR="${LR:-5e-5}"
CUTOFF_LEN="${CUTOFF_LEN:-16384}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
BASE_YAML="${BASE_YAML:-qwen3_sdpo_lora_sft.yaml}"
# DATASET overrides `dataset:` in the yaml. Default = limo_v2_sdpo (matches the
# unpatched yaml). Used by experiment 5 to plug in on-policy injection data.
DATASET="${DATASET:-limo_v2_sdpo}"

# ── Path layout ───────────────────────────────────────────────────────────────
EXP_DIR="experiments/${EXPERIMENT}"
ADAPTER_DIR="${ADAPTER_DIR:-${EXP_DIR}/adapter}"
MERGED_DIR="${MERGED_DIR:-${ADAPTER_DIR}/merged}"
RESULTS_DIR="${RESULTS_DIR:-${EXP_DIR}/eval_results}"

# ── Epistemic probe monitor ───────────────────────────────────────────────────
# eval_every_n_steps should match or divide save_steps in the training YAML
# (default save_steps=200). Lower save_steps if you want finer-grained curves.
N_PROBES="${N_PROBES:-50}"
PROBE_EVAL_STEPS="${PROBE_EVAL_STEPS:-20}"
PROBE_IDLE_MINUTES="${PROBE_IDLE_MINUTES:-30}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
SKIP_PROBE="${SKIP_PROBE:-0}"

# ── Skip flags ────────────────────────────────────────────────────────────────
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"

PY="${PYTHON:-python3}"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
echo "$EXPERIMENT" > .last_experiment

cat > "$RESULTS_DIR/train_config.txt" <<EOF
EXPERIMENT         $EXPERIMENT
RUN_ID             $RUN_ID
BASE_MODEL         $BASE_MODEL
PRETRAINED_MODEL   $PRETRAINED_MODEL
ADAPTER_DIR        $ADAPTER_DIR
MERGED_DIR         $MERGED_DIR
RESULTS_DIR        $RESULTS_DIR
N_EPOCHS           $N_EPOCHS
LORA_RANK          $LORA_RANK
LR                 $LR
CUTOFF_LEN         $CUTOFF_LEN
GRAD_ACCUM         $GRAD_ACCUM
N_PROBES           $N_PROBES
PROBE_EVAL_STEPS   $PROBE_EVAL_STEPS
WANDB_PROJECT      $WANDB_PROJECT
DATE               $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo "EXPERIMENT: $EXPERIMENT  (RUN_ID: $RUN_ID)"
echo "Config → $RESULTS_DIR/train_config.txt"

# ── Step 1: Dataset prep ──────────────────────────────────────────────────────
echo ""
echo "[1/3] LIMO-v2 dataset prep"
if [ "$SKIP_DATA" = "1" ]; then
    echo "  SKIP_DATA=1 — reusing existing limo_v2_sdpo.json"
else
    $PY make_limo_dataset.py --output limo_v2_sdpo.json
    echo "  wrote limo_v2_sdpo.json"
fi

# ── Step 2: LoRA SFT ──────────────────────────────────────────────────────────
echo ""
echo "[2/3] LoRA SFT (LlamaFactory)"
if [ "$SKIP_TRAIN" = "1" ]; then
    echo "  SKIP_TRAIN=1 — using existing adapter at $ADAPTER_DIR"
    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "ERROR: ADAPTER_DIR not found: $ADAPTER_DIR"
        exit 1
    fi
else
    PATCHED_YAML="$(mktemp /tmp/sdpo_limo_XXXXXX.yaml)"
    sed \
        -e "s|dataset_dir:.*|dataset_dir: ${HERE}|" \
        -e "s|output_dir:.*|output_dir: ${ADAPTER_DIR}|" \
        -e "s|num_train_epochs:.*|num_train_epochs: ${N_EPOCHS}|" \
        -e "s|lora_rank:.*|lora_rank: ${LORA_RANK}|" \
        -e "s|learning_rate:.*|learning_rate: ${LR}|" \
        -e "s|cutoff_len:.*|cutoff_len: ${CUTOFF_LEN}|" \
        -e "s|gradient_accumulation_steps:.*|gradient_accumulation_steps: ${GRAD_ACCUM}|" \
        -e "s|^dataset:.*|dataset: ${DATASET}|" \
        "$BASE_YAML" > "$PATCHED_YAML"
    cp "$PATCHED_YAML" "$RESULTS_DIR/training_config.yaml"
    echo "  effective config → $RESULTS_DIR/training_config.yaml"

    # ── Launch epistemic probe monitor before blocking on training ─────────────
    PROBE_CMD="$PY eval/probe_during_training.py \
        --base_model \"$BASE_MODEL\" \
        --pretrained_model \"$PRETRAINED_MODEL\" \
        --adapter_dir \"$ADAPTER_DIR\" \
        --probe_dataset limo_v2_sdpo.json \
        --n_probes $N_PROBES \
        --results_dir \"$RESULTS_DIR/epistemic_probes\" \
        --eval_every_n_steps $PROBE_EVAL_STEPS \
        --max_idle_minutes $PROBE_IDLE_MINUTES"
    if [ -n "${WANDB_PROJECT:-}" ]; then
        PROBE_CMD="$PROBE_CMD --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"${EXPERIMENT}_probe\""
    fi

    if [ "$SKIP_PROBE" = "1" ]; then
        echo "  SKIP_PROBE=1 — not starting epistemic probe monitor"
    elif [ -n "${TMUX:-}" ]; then
        echo "  tmux detected — splitting pane for epistemic probe monitor"
        tmux split-window -h "cd $HERE && eval $PROBE_CMD; echo 'Probe done. Press enter.'; read"
        echo "  probe monitor running in adjacent pane"
    else
        echo ""
        echo "  ┌─ Epistemic probe monitor ─────────────────────────────────────────┐"
        echo "  │ Open a second terminal and run:                                   │"
        echo "  │                                                                   │"
        echo "  │   cd $HERE"
        echo "  │   $PROBE_CMD"
        echo "  │                                                                   │"
        echo "  │ Or re-run with tmux for automatic pane splitting.                 │"
        echo "  └───────────────────────────────────────────────────────────────────┘"
        echo ""
    fi

    llamafactory-cli train "$PATCHED_YAML"
    rm -f "$PATCHED_YAML"
    echo "  adapter → $ADAPTER_DIR"
fi

# ── Step 3: Merge LoRA ────────────────────────────────────────────────────────
echo ""
echo "[3/3] Merging LoRA adapter"
if [ "$SKIP_MERGE" = "1" ]; then
    echo "  SKIP_MERGE=1 — skipping merge"
    MERGED_DIR=""
else
    $PY eval/merge_lora.py \
        --adapter "$ADAPTER_DIR" \
        --out "$MERGED_DIR"
    echo "  merged model → $MERGED_DIR"
fi

# ── Plot epistemic probe curves ───────────────────────────────────────────────
EPISTEMIC_DIR="$RESULTS_DIR/epistemic_probes"
LORA_SNAPSHOTS=("${EPISTEMIC_DIR}"/lora_*.json)
if [ "$SKIP_PROBE" != "1" ] && [ ${#LORA_SNAPSHOTS[@]} -gt 0 ] && [ -f "${LORA_SNAPSHOTS[0]}" ]; then
    echo ""
    echo "Plotting epistemic probe curves..."
    $PY eval/plot_probe_curves.py \
        --probe_dir "$EPISTEMIC_DIR" \
        --output "$RESULTS_DIR/probe_curves.png"
    echo "  probe curves → $RESULTS_DIR/probe_curves.png"
fi

echo ""
echo "Training complete. EXPERIMENT: $EXPERIMENT  (RUN_ID: $RUN_ID)"
echo "  adapter  → $ADAPTER_DIR"
echo "  merged   → ${MERGED_DIR:-"(skipped)"}"
echo "  probes   → $RESULTS_DIR/epistemic_probes/"
echo ""
echo "Run evaluation:"
echo "  EXPERIMENT=$EXPERIMENT ./eval.sh    # or just ./eval.sh (reads .last_experiment)"
