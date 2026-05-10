#!/usr/bin/env bash
# Experiment 7 — on-policy recovery LoRA at smaller effective batch.
#
# Same dataset as experiment 6 (~204 genuine-recovery rows from
# build_recovery_dataset.py: 78 rows from the pass-4 cohort + 126 from
# the legacy 18K-cap pass). The only training-config delta vs exp6 is
# the effective batch size: GRAD_ACCUM drops from the parent default
# of 8 to 4, doubling the number of gradient updates per epoch on the
# small cohort.
#
# Usage:
#   bash experiments/experiment7/train.sh                  # defaults
#   GRAD_ACCUM=2 bash experiments/experiment7/train.sh     # more aggressive
#   SKIP_DATASET=1 bash experiments/experiment7/train.sh   # reuse existing dataset
#   SKIP_TRAIN=1 bash experiments/experiment7/train.sh     # build dataset only
#
# Hparams: rank=8, epochs=2, lr=5e-5 (mirrors exp6). Override with
#   RANK= EPOCHS= LR= GRAD_ACCUM=

set -euo pipefail
EXP7_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP7_DIR/../.." && pwd)"
EXP6_DIR="$EXP7_DIR/../experiment6"
EXP5_DIR="$EXP7_DIR/../experiment5"
cd "$PARENT_DIR"

# ── Auto-activate .venv-train if llamafactory-cli isn't on PATH ──────────────
# The parent train.sh shells out to `llamafactory-cli`, which only lives in
# .venv-train. Skip activation if the binary is already reachable.
if ! command -v llamafactory-cli >/dev/null 2>&1; then
    VENV_TRAIN="$PARENT_DIR/.venv-train"
    if [ -f "$VENV_TRAIN/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_TRAIN/bin/activate"
        echo "  activated $VENV_TRAIN"
    else
        echo "ERROR: llamafactory-cli not on PATH and $VENV_TRAIN/bin/activate missing." >&2
        echo "       Activate the LlamaFactory venv before running this script." >&2
        exit 1
    fi
fi

EXPERIMENT="${EXPERIMENT:-experiment7}"
DATASET_PATH="${DATASET_PATH:-$EXP7_DIR/outputs/genuine_recovery_dataset.json}"
DATASET_NAME="${DATASET_NAME:-genuine_recovery_exp7}"

PASS4_TRACES="${PASS4_TRACES:-$EXP5_DIR/outputs/pass4/injection_traces.jsonl}"
LEGACY_DATASET="${LEGACY_DATASET:-$EXP5_DIR/outputs/onpolicy_recovery_dataset.json}"

# ── Hparams ──────────────────────────────────────────────────────────────────
# Rank-8 mirrors exp6 (~4× smaller cohort than LIMO's 813). The change
# in exp7 is the effective batch size: GRAD_ACCUM=4 vs the parent default 8.
RANK="${RANK:-8}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

ADAPTER_DIR="${ADAPTER_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/adapter}"
MERGED_DIR="${MERGED_DIR:-$ADAPTER_DIR/merged}"
RESULTS_DIR="${RESULTS_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/eval_results}"

SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

PY="${PYTHON:-python3}"

mkdir -p "$EXP7_DIR/outputs"

# ── Stage 1: Build dataset (reuse exp6's builder) ────────────────────────────
echo ""
echo "[1/2] Build dataset (reusing experiment 6 builder)"
if [ "$SKIP_DATASET" = "1" ] && [ -s "$DATASET_PATH" ]; then
    echo "  SKIP_DATASET=1 — reusing $DATASET_PATH"
else
    $PY "$EXP6_DIR/build_recovery_dataset.py" \
        --pass4_traces "$PASS4_TRACES" \
        --legacy_dataset "$LEGACY_DATASET" \
        --output "$DATASET_PATH" \
        --keep_provenance
fi

# Stripped training copy (LlamaFactory ignores _provenance, but we keep the
# tagged file around for analysis). Identical pattern to exp6.
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

# Register the training copy under DATASET_NAME (idempotent).
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

# ── Stage 2: LoRA SFT ────────────────────────────────────────────────────────
echo ""
echo "[2/2] LoRA SFT (delegating to parent train.sh)"
if [ "$SKIP_TRAIN" = "1" ]; then
    echo "  SKIP_TRAIN=1 — stopping before training."
    exit 0
fi

echo "[hparams] rank=$RANK epochs=$EPOCHS lr=$LR grad_accum=$GRAD_ACCUM  (exp6 used grad_accum=8)"

EXPERIMENT="$EXPERIMENT" \
ADAPTER_DIR="$ADAPTER_DIR" \
MERGED_DIR="$MERGED_DIR" \
RESULTS_DIR="$RESULTS_DIR" \
N_EPOCHS="$EPOCHS" \
LORA_RANK="$RANK" \
LR="$LR" \
GRAD_ACCUM="$GRAD_ACCUM" \
DATASET="$DATASET_NAME" \
SKIP_DATA=1 \
    bash "$PARENT_DIR/train.sh"

echo ""
echo "Experiment 7 training complete."
echo "  adapter  → $ADAPTER_DIR"
echo "  merged   → $MERGED_DIR"
echo ""
echo "Run sanity eval with:"
echo "  bash $EXP7_DIR/eval.sh"
