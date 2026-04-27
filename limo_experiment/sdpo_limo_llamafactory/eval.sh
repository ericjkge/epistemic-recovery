#!/usr/bin/env bash
# AIME evaluation → epistemic token analysis → Figure 9 alignment plots.
#
# Reads RUN_ID from the environment or from .last_run_id (written by train.sh).
# All paths are derived from RUN_ID, so no manual coordination is needed when
# running eval.sh immediately after train.sh on the same machine.
#
# Usage:
#   ./eval.sh                                       # uses RUN_ID from .last_run_id
#   RUN_ID=myrun ./eval.sh                          # explicit run ID
#   RUN_ID=myrun N_SAMPLING=16 MODELS=baseline,lora,pretrained ./eval.sh
#   RUN_ID=myrun SKIP_AIME=1 ./eval.sh             # Figure 9 only (skip AIME)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ── Resolve RUN_ID ────────────────────────────────────────────────────────────
if [ -z "${RUN_ID:-}" ]; then
    if [ -f .last_run_id ]; then
        RUN_ID="$(cat .last_run_id)"
        echo "Using RUN_ID from .last_run_id: $RUN_ID"
    else
        echo "ERROR: RUN_ID not set and .last_run_id not found."
        echo "       Run train.sh first, or set RUN_ID explicitly."
        exit 1
    fi
fi
RUN_NAME="${RUN_NAME:-sdpo_limo_lora_${RUN_ID}}"

# ── Model paths ───────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-beanie00/math-SDPO-Qwen3-8B-think-step-100}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen3-8B}"

# ── Path layout (must match train.sh) ────────────────────────────────────────
ADAPTER_DIR="${ADAPTER_DIR:-outputs/${RUN_NAME}}"
MERGED_DIR="${MERGED_DIR:-${ADAPTER_DIR}/merged}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_ID}}"

# ── AIME sampling ─────────────────────────────────────────────────────────────
N_SAMPLING="${N_SAMPLING:-4}"
MAX_TOKENS="${MAX_TOKENS:-24576}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.3}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25}"
# baseline = SDPO model without adapter, lora = SDPO + LoRA adapter
# add "pretrained" to include Qwen3-8B upper bound (slow — loads a third model)
MODELS="${MODELS:-baseline,lora}"

# ── Loop detection ────────────────────────────────────────────────────────────
LOOP_WINDOW="${LOOP_WINDOW:-300}"
LOOP_THRESHOLD="${LOOP_THRESHOLD:-0.35}"
LOOP_MIN_TOKENS="${LOOP_MIN_TOKENS:-100}"

# ── Figure 9 / epistemic alignment ───────────────────────────────────────────
N_PROBES="${N_PROBES:-50}"

# ── Skip flags ────────────────────────────────────────────────────────────────
SKIP_AIME="${SKIP_AIME:-0}"
SKIP_EPISTEMIC="${SKIP_EPISTEMIC:-0}"

PY="${PYTHON:-python3}"

# ── Sanity check ──────────────────────────────────────────────────────────────
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR not found: $RESULTS_DIR"
    echo "       Has train.sh been run for RUN_ID=$RUN_ID?"
    exit 1
fi
if [ ! -d "$ADAPTER_DIR" ] && [ ! -d "$MERGED_DIR" ]; then
    echo "ERROR: Neither ADAPTER_DIR ($ADAPTER_DIR) nor MERGED_DIR ($MERGED_DIR) found."
    exit 1
fi

cat >> "$RESULTS_DIR/train_config.txt" <<EOF

# ── Eval run appended $(date -u +%Y-%m-%dT%H:%M:%SZ) ──
N_SAMPLING         $N_SAMPLING
MAX_TOKENS         $MAX_TOKENS
BENCHMARKS         $BENCHMARKS
MODELS             $MODELS
EOF
echo "RUN_ID: $RUN_ID"

# ── Step 1: AIME evaluation ───────────────────────────────────────────────────
echo ""
echo "[1/3] AIME evaluation (n=${N_SAMPLING}, benchmarks=${BENCHMARKS})"
if [ "$SKIP_AIME" = "1" ]; then
    echo "  SKIP_AIME=1 — skipping"
else
    EVAL_ARGS=(
        --adapter_path "$ADAPTER_DIR"
        --results_dir "$RESULTS_DIR"
        --benchmarks "$BENCHMARKS"
        --models "$MODELS"
        --n_sampling "$N_SAMPLING"
        --max_tokens "$MAX_TOKENS"
        --frequency_penalty "$FREQUENCY_PENALTY"
        --loop_window "$LOOP_WINDOW"
        --loop_threshold "$LOOP_THRESHOLD"
        --loop_min_tokens "$LOOP_MIN_TOKENS"
    )
    if [ -d "${MERGED_DIR:-}" ]; then
        EVAL_ARGS+=(--lora_merged_path "$MERGED_DIR")
    fi
    $PY eval/evaluate_aime.py "${EVAL_ARGS[@]}"
fi

# ── Step 2: Epistemic token analysis ─────────────────────────────────────────
echo ""
echo "[2/3] Epistemic token analysis (count-based)"
$PY eval/analyze_epistemic.py \
    --results_dir "$RESULTS_DIR" \
    --csv_out "$RESULTS_DIR/epistemic_summary.csv" \
    --bars_out "$RESULTS_DIR/epistemic_comparison.png" \
    --scatter_out "$RESULTS_DIR/length_vs_accuracy.png" \
    --epistemic_bars_out "$RESULTS_DIR/epistemic_per_response.png" \
    --accuracy_bars_out "$RESULTS_DIR/accuracy_comparison.png"

# ── Step 3: Figure 9 — distributional alignment plots ────────────────────────
echo ""
echo "[3/3] Epistemic alignment evaluation (Figure 9)"
if [ "$SKIP_EPISTEMIC" = "1" ]; then
    echo "  SKIP_EPISTEMIC=1 — skipping"
else
    EPISTEMIC_DIR="$RESULTS_DIR/epistemic_probes"
    FIGURE9_OUT="$RESULTS_DIR/figure9.pdf"

    # Prefer probe snapshots saved by probe_during_training.py — no model loading needed.
    PROBE_JSONS=("${EPISTEMIC_DIR}"/*.json)
    if [ ${#PROBE_JSONS[@]} -gt 0 ] && [ -f "${PROBE_JSONS[0]}" ]; then
        echo "  Loading ${#PROBE_JSONS[@]} pre-computed probe snapshot(s)..."
        $PY eval/evaluate_epistemic_alignment.py \
            --from_json "${PROBE_JSONS[@]}" \
            --output "$FIGURE9_OUT"
    else
        # No snapshots — compute metrics fresh by loading models.
        echo "  No probe snapshots found; computing metrics fresh (~10 min/model)..."
        if [ -d "${MERGED_DIR:-}" ]; then
            LORA_SPEC="lora:${MERGED_DIR}"
        else
            FINAL_CKPT="$(ls -d "${ADAPTER_DIR}"/checkpoint-* 2>/dev/null \
                          | sort -t- -k2 -n | tail -1 || true)"
            LORA_SPEC="lora:${BASE_MODEL}+${FINAL_CKPT:-${ADAPTER_DIR}}"
        fi
        $PY eval/evaluate_epistemic_alignment.py \
            --probe_dataset limo_v2_sdpo.json \
            --n_probes "$N_PROBES" \
            --models \
                "pretrained:${PRETRAINED_MODEL}" \
                "sdpo:${BASE_MODEL}" \
                "${LORA_SPEC}" \
            --output "$FIGURE9_OUT"
    fi
    echo "  figure9 → $FIGURE9_OUT"
fi

echo ""
echo "Eval complete. All outputs in $RESULTS_DIR/"
echo ""
ls -1 "$RESULTS_DIR/"
