#!/usr/bin/env bash
# Sweep LoRA hyperparameters and emit per-cell train + light-eval outputs.
#
# Each cell is treated as its own EXPERIMENT under experiment4/results/<cell>/, so
# train.sh and eval.sh work unmodified. Probe snapshots (the primary signal) land
# in results/<cell>/eval_results/epistemic_probes/.
#
# Usage:
#   bash sweep.sh                                # run entire grid (idempotent)
#   FORCE=1 bash sweep.sh                        # re-train cells even if adapter exists
#   RANK=8 EPOCHS=1.0 LR=5e-5 bash sweep.sh      # single cell
#   CONFIGS_TSV=custom.tsv bash sweep.sh         # alternate grid file
#
# Cheap eval defaults: n=4 sampling, AIME25 only, no SciKnowEval, no fresh figure9
# (uses cached probe snapshots). Override per-eval bits via standard eval.sh env vars.

set -euo pipefail
EXP4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$EXP4_DIR/../.." && pwd)"     # sdpo_limo_llamafactory/
cd "$PARENT_DIR"

# ── Inputs ────────────────────────────────────────────────────────────────────
CONFIGS_TSV="${CONFIGS_TSV:-$EXP4_DIR/configs.tsv}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP4_DIR/results}"
mkdir -p "$RESULTS_ROOT"

# Light-eval defaults — keep the sweep tractable.
#
# AIME is OFF by default during the sweep: a single AIME25 pass at n=4 is
# ~30 problems × 4 samples × ~24K tokens × 8 cells ≈ multiple H100-hours just for
# tiebreak data we don't need to rank cells. The primary ranking signal is
# `recovery_pct`, which comes from the probe snapshots that train.sh's background
# probe already saves at every save_steps interval — free, no extra inference.
#
# After the sweep, run full AIME on the top 1–2 cells via `eval_winners.sh`.
SWEEP_N_SAMPLING="${SWEEP_N_SAMPLING:-4}"
SWEEP_BENCHMARKS="${SWEEP_BENCHMARKS:-aime25}"
SWEEP_MODELS="${SWEEP_MODELS:-baseline,lora}"
SWEEP_SKIP_AIME="${SWEEP_SKIP_AIME:-1}"
SWEEP_SKIP_SCIKNOWEVAL="${SWEEP_SKIP_SCIKNOWEVAL:-1}"
# Reuse cached probe snapshots: the probe runner during training already wrote
# them. evaluate_epistemic_alignment.py picks them up via --from_json.
SWEEP_SKIP_EPISTEMIC="${SWEEP_SKIP_EPISTEMIC:-0}"

FORCE="${FORCE:-0}"

# Single-cell mode — bypass the TSV.
SINGLE_CELL=0
if [ -n "${RANK:-}" ] && [ -n "${EPOCHS:-}" ] && [ -n "${LR:-}" ]; then
    SINGLE_CELL=1
fi

# ── Per-cell driver ──────────────────────────────────────────────────────────
run_cell() {
    local rank="$1" epochs="$2" lr="$3"
    # Normalize lr (5e-5 → 5e-5; tolerate user-supplied "0.00005" too) and
    # epochs (drop trailing zeros for readable directory names).
    local epochs_tag="${epochs%.0}"
    local cell_name="r${rank}_e${epochs_tag}_lr${lr}"
    local cell_dir="$RESULTS_ROOT/$cell_name"
    local exp_path="experiments/experiment4/results/$cell_name"

    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Cell: rank=$rank epochs=$epochs lr=$lr   →  $cell_name"
    echo "═══════════════════════════════════════════════════════════════════════"

    mkdir -p "$cell_dir"

    local adapter_dir="$cell_dir/adapter"
    local merged_dir="$adapter_dir/merged"
    local results_dir="$cell_dir/eval_results"

    # ── Train ────────────────────────────────────────────────────────────────
    if [ "$FORCE" != "1" ] && [ -f "$adapter_dir/adapter_model.safetensors" ]; then
        echo "[skip-train] adapter exists at $adapter_dir"
    else
        EXPERIMENT="$exp_path" \
        ADAPTER_DIR="$adapter_dir" \
        MERGED_DIR="$merged_dir" \
        RESULTS_DIR="$results_dir" \
        N_EPOCHS="$epochs" \
        LORA_RANK="$rank" \
        LR="$lr" \
        SKIP_DATA=1 \
            bash "$PARENT_DIR/train.sh"
    fi

    # ── Light eval (figure9 from cached probes; AIME skipped by default) ────
    EXPERIMENT="$exp_path" \
    ADAPTER_DIR="$adapter_dir" \
    MERGED_DIR="$merged_dir" \
    RESULTS_DIR="$results_dir" \
    N_SAMPLING="$SWEEP_N_SAMPLING" \
    BENCHMARKS="$SWEEP_BENCHMARKS" \
    MODELS="$SWEEP_MODELS" \
    SKIP_AIME="$SWEEP_SKIP_AIME" \
    SKIP_SCIKNOWEVAL="$SWEEP_SKIP_SCIKNOWEVAL" \
    SKIP_EPISTEMIC="$SWEEP_SKIP_EPISTEMIC" \
        bash "$PARENT_DIR/eval.sh"

    # Tag the cell with its hparams for the aggregator.
    cat > "$cell_dir/cell.json" <<EOF
{
  "cell": "$cell_name",
  "rank": $rank,
  "epochs": $epochs,
  "lr": "$lr"
}
EOF
    echo "[cell done] $cell_name → $cell_dir"
}

# ── Driver loop ──────────────────────────────────────────────────────────────
if [ "$SINGLE_CELL" = "1" ]; then
    run_cell "$RANK" "$EPOCHS" "$LR"
    exit 0
fi

if [ ! -f "$CONFIGS_TSV" ]; then
    echo "ERROR: configs file not found: $CONFIGS_TSV"
    exit 1
fi

# Skip header line; tolerate trailing newline / blank lines.
tail -n +2 "$CONFIGS_TSV" | while IFS=$'\t' read -r rank epochs lr; do
    [ -z "${rank:-}" ] && continue
    run_cell "$rank" "$epochs" "$lr"
done

echo ""
echo "Sweep complete. Aggregate with:"
echo "  python3 $EXP4_DIR/analyze_sweep.py --results_dir $RESULTS_ROOT --output $EXP4_DIR/summary.csv"
