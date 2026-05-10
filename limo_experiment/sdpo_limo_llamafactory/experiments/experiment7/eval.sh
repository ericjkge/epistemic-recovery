#!/usr/bin/env bash
# Sanity eval for experiment 7 — focused 7-problem AIME25 subset.
#
# Problems:
#   aime25_000..aime25_004   easy probes (Problems 1-5)
#   aime25_017               2x2 grid colorings (Problem 17, ans=82)  — case 2
#   aime25_023               sin(7*pi*sin 5x) zeros  (Problem 23, ans=149) — case 1
#
# Both case studies come from
# experiments/experiment1/eval_results/epistemic_case_studies.md.
#
# Usage:
#   bash eval.sh                                    # baseline + lora at n=4
#   N_SAMPLING=8 bash eval.sh                       # higher coverage
#   MODELS=baseline,lora,pretrained bash eval.sh    # include Qwen3-8B
#   PROBLEM_IDS=aime25_023 bash eval.sh             # one-problem probe

set -euo pipefail
EXP7_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP7_DIR/../.." && pwd)"
cd "$PARENT_DIR"

# ── Auto-activate .venv-eval if vllm isn't on PATH ───────────────────────────
# sanity_eval.py imports vllm; the `vllm` CLI exists only in .venv-eval.
# Skip activation if it's already reachable.
if ! command -v vllm >/dev/null 2>&1; then
    VENV_EVAL="$PARENT_DIR/.venv-eval"
    if [ -f "$VENV_EVAL/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_EVAL/bin/activate"
        echo "  activated $VENV_EVAL"
    else
        echo "ERROR: vllm not on PATH and $VENV_EVAL/bin/activate missing." >&2
        echo "       Activate the eval venv before running this script." >&2
        exit 1
    fi
fi

EXPERIMENT="${EXPERIMENT:-experiment7}"
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

$PY "$EXP7_DIR/sanity_eval.py" "${ARGS[@]}"

echo ""
echo "Experiment 7 sanity eval complete. Outputs in $RESULTS_DIR/"
ls -1 "$RESULTS_DIR/"
