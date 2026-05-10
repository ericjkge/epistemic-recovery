#!/usr/bin/env bash
# End-to-end driver for experiment 6 + LIMO off-policy ablation.
#
# Two modes, selected via MODE:
#   MODE=recovery (default) — merged genuine-recovery cohort
#                              (pass-4 ~78 + legacy 126 = ~204 rows).
#                              Writes to experiments/experiment6/.
#   MODE=limo               — random sample of ~200 LIMO rows (off-policy
#                              human-written solutions). Same hparams, same N.
#                              Writes to experiments/experiment6_limo/ so the
#                              two runs sit side-by-side and don't collide.
#
# Usage:
#   bash train.sh                                    # MODE=recovery (default)
#   MODE=limo bash train.sh                          # off-policy LIMO ablation
#   MODE=limo LIMO_N=200 bash train.sh               # explicit ablation size
#   MODE=limo MATCH_N=1 bash train.sh                # match exp6 row count exactly
#   RANK=8 EPOCHS=2 LR=5e-5 bash train.sh            # explicit hparams
#   DEDUPE=1 bash train.sh                           # recovery: drop duplicate questions
#   SKIP_DATASET=1 bash train.sh                     # reuse existing dataset
#   SKIP_TRAIN=1 bash train.sh                       # build dataset only, no training
#   ANALYZE_ONLY=1 bash train.sh                     # build + analyze only
#
# Hparams: mirror experiment 2's config (rank=16, epochs=2, lr=5e-5) with the
# rank dropped to 8 to match the smaller (~200-row) cohort. Override any field
# with RANK= EPOCHS= LR=.

set -euo pipefail
EXP6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP6_DIR/../.." && pwd)"     # sdpo_limo_llamafactory/
EXP4_DIR="$EXP6_DIR/../experiment4"
EXP5_DIR="$EXP6_DIR/../experiment5"
cd "$PARENT_DIR"

# ── Mode + per-mode defaults ─────────────────────────────────────────────────
MODE="${MODE:-recovery}"   # recovery | limo

case "$MODE" in
    recovery)
        DEFAULT_EXPERIMENT="experiment6"
        DEFAULT_DATASET_PATH="$EXP6_DIR/outputs/genuine_recovery_dataset.json"
        DEFAULT_DATASET_NAME="genuine_recovery"
        ;;
    limo)
        DEFAULT_EXPERIMENT="experiment6_limo"
        DEFAULT_DATASET_PATH="$EXP6_DIR/outputs/limo_ablation_dataset.json"
        DEFAULT_DATASET_NAME="limo_ablation_200"
        ;;
    *)
        echo "ERROR: MODE must be 'recovery' or 'limo' (got '$MODE')" >&2
        exit 1
        ;;
esac

# ── Stage 1 inputs (recovery mode only) ──────────────────────────────────────
PASS4_TRACES="${PASS4_TRACES:-$EXP5_DIR/outputs/pass4/injection_traces.jsonl}"
LEGACY_DATASET="${LEGACY_DATASET:-$EXP5_DIR/outputs/onpolicy_recovery_dataset.json}"

# ── Stage 1 inputs (limo mode only) ──────────────────────────────────────────
LIMO_SOURCE="${LIMO_SOURCE:-$PARENT_DIR/limo_v2_sdpo.json}"
LIMO_N="${LIMO_N:-200}"
LIMO_SEED="${LIMO_SEED:-42}"
MATCH_N="${MATCH_N:-0}"        # limo mode: match recovery row count exactly

# ── Stage 2 dataset ──────────────────────────────────────────────────────────
DATASET_PATH="${DATASET_PATH:-$DEFAULT_DATASET_PATH}"
DATASET_NAME="${DATASET_NAME:-$DEFAULT_DATASET_NAME}"
DEDUPE="${DEDUPE:-0}"          # recovery mode: 1 to drop duplicate questions

# ── Stage 3 training ─────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-$DEFAULT_EXPERIMENT}"
ADAPTER_DIR="${ADAPTER_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/adapter}"
MERGED_DIR="${MERGED_DIR:-$ADAPTER_DIR/merged}"
RESULTS_DIR="${RESULTS_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/eval_results}"

# ── Skip flags ───────────────────────────────────────────────────────────────
SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
SKIP_ANALYZE="${SKIP_ANALYZE:-0}"

PY="${PYTHON:-python3}"

# ── Resolve hparams ──────────────────────────────────────────────────────────
# Mirror experiment 2's config (rank=16, epochs=2, lr=5e-5), with rank dropped
# to 8 to match the smaller (~204-row) cohort. Each field is independently
# overridable via env vars.
resolve_hparams() {
    export RANK="${RANK:-8}"
    export EPOCHS="${EPOCHS:-2}"
    export LR="${LR:-5e-5}"
    echo "[hparams] rank=${RANK} epochs=${EPOCHS} lr=${LR}  (mirrors exp2 with rank↓ to 8)"
}

# ── Stage 1+2: Build dataset (mode-dispatched) ───────────────────────────────
echo ""
echo "[1/3] Build dataset (mode=$MODE)"
if [ "$SKIP_DATASET" = "1" ] && [ -s "$DATASET_PATH" ]; then
    echo "  SKIP_DATASET=1 — reusing $DATASET_PATH"
elif [ "$MODE" = "recovery" ]; then
    BUILD_ARGS=(
        --pass4_traces "$PASS4_TRACES"
        --legacy_dataset "$LEGACY_DATASET"
        --output "$DATASET_PATH"
        --keep_provenance
    )
    if [ "$DEDUPE" = "1" ]; then
        BUILD_ARGS+=(--dedupe_questions)
    fi
    $PY "$EXP6_DIR/build_recovery_dataset.py" "${BUILD_ARGS[@]}"
else  # MODE=limo
    BUILD_ARGS=(
        --limo_dataset "$LIMO_SOURCE"
        --n "$LIMO_N"
        --seed "$LIMO_SEED"
        --output "$DATASET_PATH"
    )
    if [ "$MATCH_N" = "1" ]; then
        BUILD_ARGS+=(--match_recovery_count)
    fi
    $PY "$EXP6_DIR/build_limo_ablation_dataset.py" "${BUILD_ARGS[@]}"
fi

# Strip _provenance into a side file LlamaFactory will ignore. We keep the
# provenance-tagged dataset for analysis but write a clean copy for training.
TRAIN_DATASET_PATH="${DATASET_PATH%.json}_train.json"
$PY - <<EOF
import json, pathlib
src = pathlib.Path("$DATASET_PATH")
dst = pathlib.Path("$TRAIN_DATASET_PATH")
rows = json.loads(src.read_text())
for r in rows:
    r.pop("_provenance", None)
dst.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(f"  wrote training copy (no provenance) → {dst}  ({len(rows)} rows)")
EOF

# Register the training copy under DATASET_NAME in dataset_info.json (idempotent).
$PY - <<EOF
import json, os, pathlib
info_path = pathlib.Path("$PARENT_DIR/dataset_info.json")
info = json.loads(info_path.read_text()) if info_path.exists() else {}
rel = os.path.relpath("$TRAIN_DATASET_PATH", "$PARENT_DIR")
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

# ── Stage 2b: Epistemic-token analysis on the SFT data ───────────────────────
echo ""
echo "[2/3] Epistemic-token analysis on SFT dataset"
if [ "$SKIP_ANALYZE" = "1" ]; then
    echo "  SKIP_ANALYZE=1 — skipping"
else
    mkdir -p "$RESULTS_DIR"
    ANALYZE_ARGS=(--dataset "$DATASET_PATH")
    if [ "$MODE" = "recovery" ]; then
        ANALYZE_ARGS+=(--by_provenance)
    fi
    $PY "$EXP6_DIR/analyze_epistemic_tokens.py" "${ANALYZE_ARGS[@]}" \
        | tee "$RESULTS_DIR/sft_dataset_epistemic.txt"
    echo "  → $RESULTS_DIR/sft_dataset_epistemic.txt"
fi

if [ "$ANALYZE_ONLY" = "1" ]; then
    echo "  ANALYZE_ONLY=1 — stopping before training."
    exit 0
fi

# ── Stage 3: LoRA SFT on the merged dataset ──────────────────────────────────
echo ""
echo "[3/3] LoRA SFT (delegating to parent train.sh)"
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
echo "Experiment 6 training complete."
echo "  adapter  → $ADAPTER_DIR"
echo "  merged   → $MERGED_DIR"
echo ""
echo "Run AIME25 evaluation with:"
echo "  bash $EXP6_DIR/eval.sh"
