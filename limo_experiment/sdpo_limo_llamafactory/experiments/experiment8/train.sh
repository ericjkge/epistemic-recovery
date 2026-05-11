#!/usr/bin/env bash
# Experiment 8 — hybrid LoRA SFT on narrativized data.
#
# Dataset (built by build_hybrid_narrative_dataset.py, on-disk at
# outputs/hybrid_narrative_dataset.json):
#   56 strict-clean recovery rows + 600 dense-short LIMO rows  → 656 total.
#   Both cohorts pass through narrativize.py so every row's <think> span is
#   pure prose: NO inline \boxed{}, NO **Final Answer** markers. The single
#   \boxed{} and "The final answer is" appear ONLY after </think>.
#
# Hparams: rank=16, epochs=2, lr=5e-5, grad_accum=8, cutoff_len=32768.
#
# Defaults to SKIP_DATASET=1: the cleaned hybrid dataset is already on disk
# at outputs/hybrid_narrative_dataset.json (the user's data-hygiene pass).
# Force a rebuild with SKIP_DATASET=0; that path needs the strict-clean
# recovery cohort at outputs/clean_recovery_dataset.json (or override
# RECOVERY_CLEAN to point at e.g. experiment 6's genuine cohort).
#
# Usage:
#   bash experiments/experiment8/train.sh                  # defaults (reuse on-disk dataset)
#   SKIP_DATASET=0 bash experiments/experiment8/train.sh   # force dataset rebuild
#   SKIP_TRAIN=1   bash experiments/experiment8/train.sh   # build dataset only
#   RANK=8 GRAD_ACCUM=4 bash experiments/experiment8/train.sh

set -euo pipefail
EXP8_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP8_DIR/../.." && pwd)"
cd "$PARENT_DIR"

# ── Auto-activate .venv-train if llamafactory-cli isn't on PATH ──────────────
if ! command -v llamafactory-cli >/dev/null 2>&1; then
    VENV_TRAIN="$PARENT_DIR/.venv-train"
    if [ -f "$VENV_TRAIN/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_TRAIN/bin/activate"
        echo "  activated $VENV_TRAIN"
    else
        echo "ERROR: llamafactory-cli not on PATH and $VENV_TRAIN/bin/activate missing." >&2
        exit 1
    fi
fi

EXPERIMENT="${EXPERIMENT:-experiment8}"
DATASET_PATH="${DATASET_PATH:-$EXP8_DIR/outputs/hybrid_narrative_dataset.json}"
DATASET_NAME="${DATASET_NAME:-hybrid_narrative_exp8}"

RECOVERY_CLEAN="${RECOVERY_CLEAN:-$EXP8_DIR/outputs/clean_recovery_dataset.json}"
LIMO_DATASET="${LIMO_DATASET:-$PARENT_DIR/limo_v2_sdpo.json}"
N_LIMO="${N_LIMO:-600}"
LENGTH_PCT_CAP="${LENGTH_PCT_CAP:-75}"
SEED="${SEED:-42}"

# ── Hparams ──────────────────────────────────────────────────────────────────
RANK="${RANK:-16}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
CUTOFF_LEN="${CUTOFF_LEN:-32768}"

ADAPTER_DIR="${ADAPTER_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/adapter}"
MERGED_DIR="${MERGED_DIR:-$ADAPTER_DIR/merged}"
RESULTS_DIR="${RESULTS_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/eval_results}"

# Default to reusing the on-disk hygiened dataset; override SKIP_DATASET=0 to rebuild.
SKIP_DATASET="${SKIP_DATASET:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

PY="${PYTHON:-python3}"

mkdir -p "$EXP8_DIR/outputs"

# ── Stage 1: build narrativized hybrid dataset (skipped by default) ─────────
echo ""
echo "[1/2] Build narrativized hybrid dataset"
if [ "$SKIP_DATASET" = "1" ] && [ -s "$DATASET_PATH" ]; then
    echo "  SKIP_DATASET=1 — reusing $DATASET_PATH"
else
    if [ ! -s "$RECOVERY_CLEAN" ]; then
        echo "ERROR: SKIP_DATASET=0 but recovery cohort missing: $RECOVERY_CLEAN" >&2
        echo "       See NOTES.md → 'Growing the clean recovery cohort' for how to" >&2
        echo "       rebuild it, or override RECOVERY_CLEAN to point at" >&2
        echo "       experiments/experiment6/outputs/genuine_recovery_dataset.json." >&2
        exit 1
    fi
    $PY "$EXP8_DIR/build_hybrid_narrative_dataset.py" \
        --recovery_clean "$RECOVERY_CLEAN" \
        --limo_dataset "$LIMO_DATASET" \
        --n_limo "$N_LIMO" \
        --length_pct_cap "$LENGTH_PCT_CAP" \
        --seed "$SEED" \
        --output "$DATASET_PATH" \
        --strict
fi

# Stripped training copy (LlamaFactory ignores _provenance, but we keep the
# tagged file around for analysis). Identical pattern to exp 7.
TRAIN_DATASET_PATH="${DATASET_PATH%.json}_train.json"
$PY - <<EOF
import json, pathlib
src = pathlib.Path("$DATASET_PATH")
dst = pathlib.Path("$TRAIN_DATASET_PATH")
rows = json.loads(src.read_text())
for r in rows:
    r.pop("_provenance", None)
    r.pop("_phrase_rewritten", None)
dst.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
print(f"  wrote training copy → {dst}  ({len(rows)} rows)")
EOF

# Register the training copy under DATASET_NAME in dataset_info.json.
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
print(f"  registered '$DATASET_NAME' → {rel}")
EOF

# ── Stage 2: LoRA SFT ────────────────────────────────────────────────────────
echo ""
echo "[2/2] LoRA SFT (delegating to parent train.sh)"
if [ "$SKIP_TRAIN" = "1" ]; then
    echo "  SKIP_TRAIN=1 — stopping before training."
    exit 0
fi

echo "[hparams] rank=$RANK epochs=$EPOCHS lr=$LR grad_accum=$GRAD_ACCUM cutoff=$CUTOFF_LEN"

EXPERIMENT="$EXPERIMENT" \
ADAPTER_DIR="$ADAPTER_DIR" \
MERGED_DIR="$MERGED_DIR" \
RESULTS_DIR="$RESULTS_DIR" \
N_EPOCHS="$EPOCHS" \
LORA_RANK="$RANK" \
LR="$LR" \
GRAD_ACCUM="$GRAD_ACCUM" \
CUTOFF_LEN="$CUTOFF_LEN" \
DATASET="$DATASET_NAME" \
SKIP_DATA=1 \
    bash "$PARENT_DIR/train.sh"

echo ""
echo "Experiment 8 training complete."
echo "  adapter  → $ADAPTER_DIR"
echo "  merged   → $MERGED_DIR"
echo ""
echo "Run sanity eval with:"
echo "  bash $EXP8_DIR/eval.sh"
