#!/usr/bin/env bash
# AIME25-only evaluation for experiment 6 + epistemic token analysis.
#
# Delegates to the parent eval.sh with BENCHMARKS=aime25 and skips the
# SciKnowEval / Figure-9 stages (the parent script's count-based epistemic
# analysis is kept — that's the per-token table over generations).
#
# Also re-runs the SFT-data epistemic analysis so the eval_results/ folder
# is self-contained.
#
# Usage:
#   bash eval.sh                                  # AIME25 + epistemic analysis
#   N_SAMPLING=8 bash eval.sh                     # higher pass@k coverage
#   MODELS=baseline,lora,pretrained bash eval.sh  # include Qwen3-8B upper bound
#   SKIP_AIME=1 bash eval.sh                      # epistemic analysis only

set -euo pipefail
EXP6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP6_DIR/../.." && pwd)"
cd "$PARENT_DIR"

EXPERIMENT="${EXPERIMENT:-experiment6}"
RESULTS_DIR="${RESULTS_DIR:-$PARENT_DIR/experiments/${EXPERIMENT}/eval_results}"
DATASET_PATH="${DATASET_PATH:-$EXP6_DIR/outputs/genuine_recovery_dataset.json}"

# AIME25 only — exp6's training cohort is small and the comparison frame for
# this run is "did 203 genuine-recovery rows shift AIME25 vs baseline / vs
# the 813-row LIMO LoRA?" AIME24 can be added back via BENCHMARKS=aime24,aime25.
BENCHMARKS="${BENCHMARKS:-aime25}"
N_SAMPLING="${N_SAMPLING:-4}"
MODELS="${MODELS:-baseline,lora}"
MAX_TOKENS="${MAX_TOKENS:-24576}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-28672}"

# Stages we don't want for this eval.
SKIP_SCIKNOWEVAL="${SKIP_SCIKNOWEVAL:-1}"
SKIP_EPISTEMIC="${SKIP_EPISTEMIC:-1}"   # parent's Figure 9 alignment plot
SKIP_AIME="${SKIP_AIME:-0}"
SKIP_DATASET_EPISTEMIC="${SKIP_DATASET_EPISTEMIC:-0}"

PY="${PYTHON:-python3}"

mkdir -p "$RESULTS_DIR"

# ── Step A: SFT-data epistemic-token snapshot ────────────────────────────────
echo ""
echo "[A] Epistemic-token analysis on SFT dataset"
if [ "$SKIP_DATASET_EPISTEMIC" = "1" ]; then
    echo "  SKIP_DATASET_EPISTEMIC=1 — skipping"
elif [ ! -s "$DATASET_PATH" ]; then
    echo "  WARNING: $DATASET_PATH not found — skipping (run train.sh first)"
else
    $PY "$EXP6_DIR/analyze_epistemic_tokens.py" \
        --dataset "$DATASET_PATH" \
        --by_provenance \
        | tee "$RESULTS_DIR/sft_dataset_epistemic.txt"
fi

# ── Step B: Delegate to parent eval.sh for AIME + per-generation analysis ───
echo ""
echo "[B] AIME${BENCHMARKS//,/+} (n=${N_SAMPLING}, models=${MODELS})"
EXPERIMENT="$EXPERIMENT" \
RESULTS_DIR="$RESULTS_DIR" \
BENCHMARKS="$BENCHMARKS" \
N_SAMPLING="$N_SAMPLING" \
MODELS="$MODELS" \
MAX_TOKENS="$MAX_TOKENS" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
SKIP_AIME="$SKIP_AIME" \
SKIP_SCIKNOWEVAL="$SKIP_SCIKNOWEVAL" \
SKIP_EPISTEMIC="$SKIP_EPISTEMIC" \
    bash "$PARENT_DIR/eval.sh"

echo ""
echo "Experiment 6 eval complete. Outputs in $RESULTS_DIR/"
