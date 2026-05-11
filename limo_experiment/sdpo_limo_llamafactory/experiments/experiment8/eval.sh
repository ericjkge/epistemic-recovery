#!/usr/bin/env bash
# Sanity eval for experiment 8 — same 7-problem AIME25 subset as exp 7.
set -euo pipefail
EXP8_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP8_DIR/../.." && pwd)"
cd "$PARENT_DIR"

if ! command -v vllm >/dev/null 2>&1; then
    VENV_EVAL="$PARENT_DIR/.venv-eval"
    if [ -f "$VENV_EVAL/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_EVAL/bin/activate"
        echo "  activated $VENV_EVAL"
    else
        echo "ERROR: vllm not on PATH and $VENV_EVAL/bin/activate missing." >&2
        exit 1
    fi
fi

EXPERIMENT="${EXPERIMENT:-experiment8}"
ADAPTER_DIR="${ADAPTER_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/adapter}"
MERGED_DIR="${MERGED_DIR:-$ADAPTER_DIR/merged}"
RESULTS_DIR="${RESULTS_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/eval_results}"

N_SAMPLING="${N_SAMPLING:-4}"
MODELS="${MODELS:-baseline,lora}"
MAX_TOKENS="${MAX_TOKENS:-24576}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-28672}"
PROBLEM_IDS="${PROBLEM_IDS:-aime25_000,aime25_001,aime25_002,aime25_003,aime25_004,aime25_017,aime25_023}"

PY="${PYTHON:-python3}"
mkdir -p "$RESULTS_DIR"

echo "EXPERIMENT: $EXPERIMENT"
echo "  ADAPTER_DIR  $ADAPTER_DIR"
echo "  MERGED_DIR   $MERGED_DIR"
echo "  RESULTS_DIR  $RESULTS_DIR"
echo "  problems     $PROBLEM_IDS"
echo "  models       $MODELS"
echo "  n_sampling   $N_SAMPLING"

if [ ! -d "$ADAPTER_DIR" ] && [ ! -d "$MERGED_DIR" ]; then
    echo "ERROR: Neither ADAPTER_DIR ($ADAPTER_DIR) nor MERGED_DIR ($MERGED_DIR) exists."
    echo "       Run train.sh first."
    exit 1
fi

ARGS=(
    --adapter_path "$ADAPTER_DIR"
    --results_dir "$RESULTS_DIR"
    --problem_ids "$PROBLEM_IDS"
    --models "$MODELS"
    --n_sampling "$N_SAMPLING"
    --max_tokens "$MAX_TOKENS"
    --max_model_len "$MAX_MODEL_LEN"
)
if [ -d "${MERGED_DIR:-}" ]; then
    ARGS+=(--lora_merged_path "$MERGED_DIR")
fi

$PY "$EXP8_DIR/sanity_eval.py" "${ARGS[@]}"

echo ""
echo "Experiment 8 sanity eval complete. Outputs in $RESULTS_DIR/"
ls -1 "$RESULTS_DIR/"
