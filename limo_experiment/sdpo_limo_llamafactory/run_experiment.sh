#!/usr/bin/env bash
# End-to-end pipeline: data → train LoRA → evaluate 3 models → epistemic analysis.
# Run from sdpo_limo_llamafactory/.

set -euo pipefail

# Pin to the right interpreter (user reported python3.11 in the active venv).
PY="${PYTHON:-python3.11}"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

CONFIG="qwen3_sdpo_lora_sft.yaml"
ADAPTER_DIR="saves/qwen3_sdpo_limo_lora"

echo "[1/4] Reshape LIMO into Qwen3 thinking-mode format"
$PY prepare_limo.py \
  --dataset GAIR/LIMO \
  --output data/limo_qwen3_thinking.json \
  --tokenizer_name Qwen/Qwen3-8B

echo "[2/4] LoRA SFT (LlamaFactory)"
# llamafactory-cli is installed by `pip install -e .[torch,metrics]` in LLaMA-Factory/.
llamafactory-cli train "$CONFIG"

echo "[3/4] Evaluate baseline / lora / pretrained on AIME24 + AIME25 (vLLM, n=16)"
$PY evaluate_aime.py \
  --adapter_path "$ADAPTER_DIR" \
  --results_dir results \
  --benchmarks aime24,aime25 \
  --models baseline,lora,pretrained \
  --n_sampling 16 \
  --max_tokens 38912

echo "[4/4] Epistemic-token analysis"
$PY analyze_epistemic.py --results_dir results

echo "Done. See results/epistemic_summary.csv, results/epistemic_comparison.png, results/length_vs_accuracy.png"
