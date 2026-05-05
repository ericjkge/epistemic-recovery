#!/usr/bin/env bash
# MI-peak comparison on AIME25 problem 23 (LoRA-only-solved, see experiment2/NOTES.md).
#
# Requires GPU. The merged LoRA weights at experiment2/adapter/merged/ are missing the
# safetensor shards locally — re-merge with eval/merge_lora.py before running, or point
# LORA_MODEL at a path that has full weights.

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
CALC="$(realpath "$EXP_DIR/../../../strategic-information-allocation-llm-reasoning/mi_peak/calc.py")"

LORA_MODEL="${LORA_MODEL:-$EXP_DIR/../experiment2/adapter/merged}"
BASE_MODEL="${BASE_MODEL:-beanie00/math-SDPO-Qwen3-8B-think-step-100}"

# Qwen3-8B has 36 layers; calc.py uses layer (num_layers + idx + 1) for negative idx.
# MI-Peaks paper uses a late layer; plot_mi.sh uses layer 36 for Qwen3-8B.
LAYER="${LAYER:-36}"
MAX_TOKENS="${MAX_TOKENS:-4000}"   # baseline trace ~10k chars, lora ~30k chars

run() {
    local label="$1" model="$2" input="$3"
    echo "=== $label ==="
    python3 "$CALC" \
        --model_path "$model" \
        --input_file "$EXP_DIR/inputs/$input" \
        --indices "0-0" \
        --max_tokens "$MAX_TOKENS" \
        --layer "$LAYER" \
        --output_dir "$EXP_DIR/results/$label"
}

run lora_correct "$LORA_MODEL" aime23_lora_correct.jsonl
run base_wrong  "$BASE_MODEL" aime23_base_wrong.jsonl

echo "Done. CSVs under $EXP_DIR/results/"
echo "Plot with: python3 ../../../strategic-information-allocation-llm-reasoning/mi_peak/plot_mi.py \\"
echo "    --csv  $EXP_DIR/results/lora_correct/mi_first${MAX_TOKENS}_layer${LAYER}_answer_only.csv \\"
echo "    --csv2 $EXP_DIR/results/base_wrong/mi_first${MAX_TOKENS}_layer${LAYER}_answer_only.csv \\"
echo "    --title 'LoRA correct (149)' --title2 'Baseline wrong (150)' --output $EXP_DIR/results/aime23"
