#!/usr/bin/env bash
# End-to-end driver for experiment 5: seed prep → injection generation →
# dataset construction → LoRA SFT on the on-policy dataset.
#
# Stages are skip-flagged so you can re-enter at any point.
#
# Usage:
#   bash train.sh                                       # full pipeline, hparams from best_config.json
#   RANK=8 EPOCHS=1.0 LR=5e-5 bash train.sh             # explicit hparams (overrides best_config)
#   POLICY=baseline bash train.sh                       # generate from raw SDPO instead of exp4 LoRA
#   SKIP_GEN=1 bash train.sh                            # reuse existing injection_traces.jsonl
#   SKIP_DATASET=1 SKIP_GEN=1 bash train.sh             # reuse existing onpolicy_injection_dataset.json
#   SKIP_TRAIN=1 bash train.sh                          # only refresh seeds/dataset
#
# Hparams resolution order (first wins):
#   1. RANK / EPOCHS / LR env vars
#   2. ../experiment4/best_config.json {"rank": N, "epochs": F, "lr": "Xe-Y"}
#   3. exp1/2 defaults: rank=16, epochs=2, lr=5e-5  (fallback only — flag a warning)

set -euo pipefail
EXP5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP5_DIR/../.." && pwd)"     # sdpo_limo_llamafactory/
EXP4_DIR="$EXP5_DIR/../experiment4"
cd "$PARENT_DIR"

# ── Stage 1 inputs ───────────────────────────────────────────────────────────
# Default: LIMO questions — same set the off-policy LIMO LoRA (exp1/2/4) was
# trained on. This makes experiment 5 a clean on-policy/off-policy ablation
# on a matched question distribution. AIME24/25 are not in LIMO, so the held-
# out eval stays disjoint.
SEED_SOURCE="${SEED_SOURCE:-GAIR/LIMO-v2}"
SEED_MAX="${SEED_MAX:-}"     # empty = take all integer-answer LIMO questions
SEEDS_PATH="${SEEDS_PATH:-$EXP5_DIR/seed_problems.jsonl}"

# ── Stage 2 generation ───────────────────────────────────────────────────────
POLICY="${POLICY:-lora}"          # baseline | lora | pretrained
EXP4_WINNER="${EXP4_WINNER:-}"    # e.g. r8_e1_lr5e-5  (cell name under exp4/results/)
MAX_ROUNDS="${MAX_ROUNDS:-3}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-12288}"
CUMULATIVE_TOKEN_CAP="${CUMULATIVE_TOKEN_CAP:-22528}"
TRACES_PATH="${TRACES_PATH:-$EXP5_DIR/outputs/injection_traces.jsonl}"

# ── Stage 3 dataset ──────────────────────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-$EXP5_DIR/outputs/onpolicy_injection_dataset.json}"
DATASET_NAME="${DATASET_NAME:-onpolicy_injection}"
REQUIRE_INJECTION="${REQUIRE_INJECTION:-0}"

# ── Stage 4 training ─────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-experiment5}"
ADAPTER_DIR="${ADAPTER_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/adapter}"
MERGED_DIR="${MERGED_DIR:-$ADAPTER_DIR/merged}"
RESULTS_DIR="${RESULTS_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/eval_results}"

# ── Skip flags ───────────────────────────────────────────────────────────────
SKIP_SEEDS="${SKIP_SEEDS:-0}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

PY="${PYTHON:-python3}"

# ── Resolve hparams from exp4 best_config.json ───────────────────────────────
resolve_hparams() {
    local rank epochs lr
    if [ -n "${RANK:-}" ] && [ -n "${EPOCHS:-}" ] && [ -n "${LR:-}" ]; then
        echo "[hparams] using env vars: rank=${RANK} epochs=${EPOCHS} lr=${LR}"
        return
    fi
    local cfg="$EXP4_DIR/best_config.json"
    if [ -f "$cfg" ]; then
        rank=$($PY -c "import json; print(json.load(open('$cfg'))['rank'])")
        epochs=$($PY -c "import json; print(json.load(open('$cfg'))['epochs'])")
        lr=$($PY -c "import json; print(json.load(open('$cfg'))['lr'])")
        export RANK="$rank" EPOCHS="$epochs" LR="$lr"
        echo "[hparams] from exp4/best_config.json: rank=${RANK} epochs=${EPOCHS} lr=${LR}"
        return
    fi
    export RANK="${RANK:-16}"
    export EPOCHS="${EPOCHS:-2}"
    export LR="${LR:-5e-5}"
    echo "[hparams] WARNING: no exp4/best_config.json found — falling back to exp1/2 defaults"
    echo "[hparams]          rank=${RANK} epochs=${EPOCHS} lr=${LR}"
    echo "[hparams]          run experiment 4 first, then re-run this script."
}

# ── Resolve exp4 winning policy for generation (POLICY=lora) ─────────────────
resolve_policy_args() {
    POLICY_ARGS=()
    if [ "$POLICY" != "lora" ]; then
        return
    fi
    local cell="$EXP4_WINNER"
    if [ -z "$cell" ] && [ -f "$EXP4_DIR/best_config.json" ]; then
        cell=$($PY -c "import json; d=json.load(open('$EXP4_DIR/best_config.json')); print(d.get('cell',''))")
    fi
    if [ -n "$cell" ] && [ -d "$EXP4_DIR/results/$cell" ]; then
        local merged="$EXP4_DIR/results/$cell/adapter/merged"
        local adapter="$EXP4_DIR/results/$cell/adapter"
        if [ -f "$merged/config.json" ]; then
            POLICY_ARGS=(--merged "$merged")
            echo "[policy] using exp4 cell=$cell merged checkpoint"
            return
        fi
        if [ -d "$adapter" ]; then
            POLICY_ARGS=(--adapter "$adapter")
            echo "[policy] using exp4 cell=$cell adapter (no merged dir found)"
            return
        fi
    fi
    # Fall back to exp2's adapter if exp4 hasn't run.
    if [ -d "$PARENT_DIR/experiments/experiment2/adapter/merged" ] \
       && [ -f "$PARENT_DIR/experiments/experiment2/adapter/merged/config.json" ]; then
        POLICY_ARGS=(--merged "$PARENT_DIR/experiments/experiment2/adapter/merged")
        echo "[policy] WARNING: no exp4 winner found; falling back to experiment2 LoRA"
        return
    fi
    echo "[policy] ERROR: --model lora requested but no merged/adapter checkpoint found."
    echo "                Set EXP4_WINNER=<cell> or POLICY=baseline to use raw SDPO."
    exit 1
}

# ── Stage 1: Seed problems ───────────────────────────────────────────────────
echo ""
echo "[1/4] Seed problems"
if [ "$SKIP_SEEDS" = "1" ] || [ -s "$SEEDS_PATH" ]; then
    if [ -s "$SEEDS_PATH" ]; then
        echo "  reusing existing $SEEDS_PATH ($(wc -l < "$SEEDS_PATH") rows)"
    else
        echo "  SKIP_SEEDS=1 but $SEEDS_PATH is empty/missing — re-run without skip"
        exit 1
    fi
else
    PREP_ARGS=(--source "$SEED_SOURCE" --output "$SEEDS_PATH")
    if [ -n "$SEED_MAX" ]; then
        PREP_ARGS+=(--max "$SEED_MAX")
    fi
    $PY "$EXP5_DIR/prepare_seed_problems.py" "${PREP_ARGS[@]}"
fi

# ── Stage 2: Iterative injection generation ──────────────────────────────────
echo ""
echo "[2/4] Iterative injection generation (policy=$POLICY)"
if [ "$SKIP_GEN" = "1" ] && [ -s "$TRACES_PATH" ]; then
    echo "  SKIP_GEN=1 — reusing $TRACES_PATH ($(wc -l < "$TRACES_PATH") rows)"
else
    resolve_policy_args
    $PY "$EXP5_DIR/generate_injections.py" \
        --seeds "$SEEDS_PATH" \
        --output_dir "$EXP5_DIR/outputs" \
        --model "$POLICY" \
        --max_rounds "$MAX_ROUNDS" \
        --max_tokens "$GEN_MAX_TOKENS" \
        --cumulative_token_cap "$CUMULATIVE_TOKEN_CAP" \
        "${POLICY_ARGS[@]}"
fi

# ── Stage 3: Dataset construction ────────────────────────────────────────────
echo ""
echo "[3/4] Dataset construction → $DATASET_PATH"
if [ "$SKIP_DATASET" = "1" ] && [ -s "$DATASET_PATH" ]; then
    echo "  SKIP_DATASET=1 — reusing $DATASET_PATH"
else
    extra=""
    if [ "$REQUIRE_INJECTION" = "1" ]; then extra="--require_injection"; fi
    $PY "$EXP5_DIR/make_injection_dataset.py" \
        --traces "$TRACES_PATH" \
        --output "$DATASET_PATH" \
        $extra
fi

# Register the dataset under DATASET_NAME in dataset_info.json (idempotent).
$PY - <<EOF
import json, os, pathlib
info_path = pathlib.Path("$PARENT_DIR/dataset_info.json")
info = json.loads(info_path.read_text()) if info_path.exists() else {}
rel = os.path.relpath("$DATASET_PATH", "$PARENT_DIR")
info["$DATASET_NAME"] = {
    "file_name": rel,
    "formatting": "alpaca",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
    },
}
info_path.write_text(json.dumps(info, indent=2) + "\n")
print(f"  registered '$DATASET_NAME' → {rel} in {info_path}")
EOF

# ── Stage 4: LoRA SFT on the on-policy dataset ───────────────────────────────
echo ""
echo "[4/4] LoRA SFT (delegating to parent train.sh)"
if [ "$SKIP_TRAIN" = "1" ]; then
    echo "  SKIP_TRAIN=1 — stopping before training."
    exit 0
fi

resolve_hparams

EXPERIMENT="$EXPERIMENT" \
ADAPTER_DIR="$ADAPTER_DIR" \
MERGED_DIR="$MERGED_DIR" \
RESULTS_DIR="$RESULTS_DIR" \
N_EPOCHS="$EPOCHS" \
LORA_RANK="$RANK" \
LR="$LR" \
DATASET="$DATASET_NAME" \
SKIP_DATA=1 \
    bash "$PARENT_DIR/train.sh"

echo ""
echo "Experiment 5 training complete."
echo "  adapter  → $ADAPTER_DIR"
echo "  merged   → $MERGED_DIR"
echo ""
echo "Run evaluation with:"
echo "  EXPERIMENT=$EXPERIMENT bash $PARENT_DIR/eval.sh"
