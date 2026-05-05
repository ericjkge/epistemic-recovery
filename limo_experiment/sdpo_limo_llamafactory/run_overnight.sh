#!/usr/bin/env bash
# Overnight driver: experiment 4 sweep -> per-cell probe -> analyze -> exp 5 pipeline.
#
# Probes are run SEQUENTIALLY (post-training, one snapshot per cell) instead of
# in a parallel tmux pane — the parallel mode OOMs the H100 with cutoff_len=16384
# because the probe holds two reference models (~32GB) while the trainer needs
# ~37GB for activations.
#
# Logs to experiments/overnight.log. Touches experiments/OVERNIGHT_DONE on success
# and experiments/OVERNIGHT_FAILED with the failing stage on any error.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

LOG="$HERE/experiments/overnight.log"
DONE_MARKER="$HERE/experiments/OVERNIGHT_DONE"
FAIL_MARKER="$HERE/experiments/OVERNIGHT_FAILED"
mkdir -p "$HERE/experiments"
rm -f "$DONE_MARKER" "$FAIL_MARKER"

VENV_TRAIN="$HERE/.venv-train"
VENV_EVAL="$HERE/.venv-eval"

BASE_MODEL="${BASE_MODEL:-beanie00/math-SDPO-Qwen3-8B-think-step-100}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen3-8B}"
N_PROBES="${N_PROBES:-50}"

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*"; }

fail() {
    local stage="$1" msg="${2:-}"
    say "FAIL [$stage] $msg"
    echo "$stage" > "$FAIL_MARKER"
    exit 1
}

run_stage() {
    local name="$1"; shift
    say "BEGIN $name"
    say "  cmd: $*"
    if "$@"; then
        say "END   $name (ok)"
    else
        local rc=$?
        fail "$name" "rc=$rc"
    fi
}

probe_cell() {
    # Run a single-snapshot probe on a cell's final adapter. Skips the heavy
    # baseline pass (we rely on experiment2's cached refs via analyze_sweep's
    # fallback). Picks the latest checkpoint-N subdir if any, else the adapter
    # dir itself.
    local cell_dir="$1"
    local adapter_dir="$cell_dir/adapter"
    local probe_dir="$cell_dir/eval_results/epistemic_probes"

    if [ ! -f "$adapter_dir/adapter_model.safetensors" ]; then
        say "  [probe] skipping — no adapter at $adapter_dir"
        return 0
    fi

    mkdir -p "$probe_dir"
    if compgen -G "$probe_dir/lora_step*.json" > /dev/null; then
        say "  [probe] already present in $probe_dir — skipping"
        return 0
    fi

    local ckpt
    ckpt=$(ls -d "$adapter_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$ckpt" ]; then
        ckpt="$adapter_dir"
    fi
    say "  [probe] $cell_dir  ckpt=$ckpt"

    if python3 "$HERE/eval/probe_during_training.py" \
        --base_model "$BASE_MODEL" \
        --pretrained_model "$PRETRAINED_MODEL" \
        --adapter_dir "$adapter_dir" \
        --probe_dataset "$HERE/limo_v2_sdpo.json" \
        --n_probes "$N_PROBES" \
        --results_dir "$probe_dir" \
        --skip_baselines \
        --final_checkpoint "$ckpt" >> "$LOG" 2>&1; then
        say "  [probe] ok"
    else
        fail "probe_$cell_dir" "probe failed; see $LOG"
    fi
}

say "================================================================"
say "Overnight run starting"
say "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
say "================================================================"

# ── Stage A: experiment 4 sweep (probes deferred to per-cell post-pass) ─────
say "Activating .venv-train"
# shellcheck disable=SC1091
source "$VENV_TRAIN/bin/activate"
say "  python=$(which python)"
say "  llamafactory-cli=$(which llamafactory-cli || echo MISSING)"

export SKIP_PROBE=1   # propagates into sweep.sh -> train.sh
run_stage "exp4_sweep" bash "$HERE/experiments/experiment4/sweep.sh"
unset SKIP_PROBE

# ── Stage B: per-cell post-training probe ───────────────────────────────────
say "BEGIN exp4_probes"
for cell_dir in "$HERE"/experiments/experiment4/results/*/; do
    cell_dir="${cell_dir%/}"
    [ -f "$cell_dir/cell.json" ] || continue
    probe_cell "$cell_dir"
done
say "END   exp4_probes (ok)"

# ── Stage C: analyze sweep ──────────────────────────────────────────────────
run_stage "exp4_analyze" python3 "$HERE/experiments/experiment4/analyze_sweep.py" \
    --results_dir "$HERE/experiments/experiment4/results" \
    --output "$HERE/experiments/experiment4/summary.csv"

if [ ! -f "$HERE/experiments/experiment4/best_config.json" ]; then
    fail "exp4_best_config" "best_config.json missing after analyze_sweep.py"
fi
say "exp4 winner: $(cat "$HERE/experiments/experiment4/best_config.json")"

deactivate || true

# ── Stage D: experiment 5 stages 1-3 (need vllm for generation) ─────────────
say "Activating .venv-eval for exp5 generation"
# shellcheck disable=SC1091
source "$VENV_EVAL/bin/activate"
say "  python=$(which python)"
python -c "import vllm; print('  vllm', vllm.__version__)" || fail "exp5_vllm_check" "vllm import failed"

# SEED_MAX=200 for first end-to-end run (per NOTES.md "during pipeline shake-out").
# POLICY=lora reads exp4's best_config.json to pick the generation adapter.
run_stage "exp5_gen" env \
    SEED_MAX=200 \
    POLICY=lora \
    SKIP_TRAIN=1 \
    PYTHON="$VENV_EVAL/bin/python" \
    bash "$HERE/experiments/experiment5/train.sh"

deactivate || true

# ── Stage E: experiment 5 stage 4 (LoRA SFT, needs .venv-train) ─────────────
say "Re-activating .venv-train for exp5 training"
# shellcheck disable=SC1091
source "$VENV_TRAIN/bin/activate"

# Skip probe here too — exp5 just needs the trained adapter; we can probe later.
export SKIP_PROBE=1
run_stage "exp5_train" env \
    SKIP_SEEDS=1 \
    SKIP_GEN=1 \
    SKIP_DATASET=1 \
    bash "$HERE/experiments/experiment5/train.sh"
unset SKIP_PROBE

deactivate || true

say "================================================================"
say "Overnight run complete"
say "================================================================"
touch "$DONE_MARKER"
