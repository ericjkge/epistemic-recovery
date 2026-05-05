#!/usr/bin/env bash
# Run full AIME eval on the top-ranked sweep cells.
#
# `sweep.sh` skips AIME by default (recovery_pct alone is enough to rank cells).
# This script reads `summary.csv` (or a comma-separated list) and runs full AIME
# (n=8, both AIME24 + AIME25) on the top K cells — that's the eval budget that
# matters once you've narrowed the field.
#
# Usage:
#   bash eval_winners.sh                            # top 2 cells from summary.csv
#   TOP_K=3 bash eval_winners.sh                    # top 3
#   CELLS=r8_e1_lr5e-5,r4_e1_lr5e-5 bash eval_winners.sh   # explicit list
#   N_SAMPLING=4 BENCHMARKS=aime25 bash eval_winners.sh    # cheaper override

set -euo pipefail
EXP4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP4_DIR/../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP4_DIR/results}"
SUMMARY="${SUMMARY:-$EXP4_DIR/summary.csv}"
TOP_K="${TOP_K:-2}"

# Eval depth — full numbers by default, mirroring experiment 2's rigor.
N_SAMPLING="${N_SAMPLING:-8}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25}"
MODELS="${MODELS:-baseline,lora}"
SKIP_SCIKNOWEVAL="${SKIP_SCIKNOWEVAL:-1}"

# ── Resolve cell list ────────────────────────────────────────────────────────
if [ -n "${CELLS:-}" ]; then
    IFS=',' read -ra CELL_LIST <<< "$CELLS"
else
    if [ ! -f "$SUMMARY" ]; then
        echo "ERROR: $SUMMARY not found. Run analyze_sweep.py first."
        exit 1
    fi
    # summary.csv is already sorted by |recovery_pct − 1|; cell is column 1.
    mapfile -t CELL_LIST < <(tail -n +2 "$SUMMARY" | head -n "$TOP_K" | cut -d',' -f1)
fi

echo "Evaluating $TOP_K winner(s):"
printf '  %s\n' "${CELL_LIST[@]}"

cd "$PARENT_DIR"
for cell in "${CELL_LIST[@]}"; do
    cell_dir="$RESULTS_ROOT/$cell"
    if [ ! -d "$cell_dir/adapter" ]; then
        echo "[skip] $cell — no adapter at $cell_dir/adapter"
        continue
    fi
    echo ""
    echo "═══ $cell ═══"
    EXPERIMENT="experiments/experiment4/results/$cell" \
    ADAPTER_DIR="$cell_dir/adapter" \
    MERGED_DIR="$cell_dir/adapter/merged" \
    RESULTS_DIR="$cell_dir/eval_results" \
    N_SAMPLING="$N_SAMPLING" \
    BENCHMARKS="$BENCHMARKS" \
    MODELS="$MODELS" \
    SKIP_SCIKNOWEVAL="$SKIP_SCIKNOWEVAL" \
        bash "$PARENT_DIR/eval.sh"
done

echo ""
echo "Done. Re-run analyze_sweep.py to refresh summary.csv with the new AIME numbers:"
echo "  python3 $EXP4_DIR/analyze_sweep.py --results_dir $RESULTS_ROOT --output $EXP4_DIR/summary.csv"
