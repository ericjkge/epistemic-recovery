#!/bin/bash
# End-to-end pipeline: LIMO-v2 dataset prep → LlamaFactory LoRA SFT → AIME eval
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="${SCRIPT_DIR}/../strategic-information-allocation-llm-reasoning/train"
YAML="${SCRIPT_DIR}/qwen3_sdpo_lora_sft.yaml"
ADAPTER_DIR="${SCRIPT_DIR}/outputs/sdpo-qwen3-8b-limo-lora-r16"

# ── Step 1: Generate dataset ───────────────────────────────────────────────
echo "=== Step 1: Generate LIMOv2 training data ==="
python "${SCRIPT_DIR}/make_limo_dataset.py" \
    --output "${SCRIPT_DIR}/limo_v2_sdpo.json"

# ── Step 2: Patch dataset_dir in YAML ─────────────────────────────────────
echo ""
echo "=== Step 2: Patch dataset_dir in YAML ==="
# Replace the REPLACE_WITH_ABSOLUTE_PATH placeholder with the actual path
sed -i.bak "s|REPLACE_WITH_ABSOLUTE_PATH|${SCRIPT_DIR}|g" "${YAML}"
echo "  dataset_dir set to: ${SCRIPT_DIR}"

# ── Step 3: LlamaFactory training ─────────────────────────────────────────
echo ""
echo "=== Step 3: LlamaFactory LoRA SFT ==="
if [ ! -d "${LLAMA_FACTORY_DIR}" ]; then
    echo "ERROR: LlamaFactory not found at ${LLAMA_FACTORY_DIR}"
    echo "  Install with:  pip install llamafactory"
    echo "  Or clone into: ${LLAMA_FACTORY_DIR}"
    exit 1
fi

# Single GPU
llamafactory-cli train "${YAML}"

# For multi-GPU, replace the above with:
# N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# torchrun --nproc_per_node="${N_GPUS}" \
#     "${LLAMA_FACTORY_DIR}/src/train.py" "${YAML}"

# ── Step 4: AIME + epistemic evaluation ───────────────────────────────────
echo ""
echo "=== Step 4: AIME + epistemic evaluation ==="
python "${SCRIPT_DIR}/eval_aime_epistemic.py" \
    --adapter_path "${ADAPTER_DIR}" \
    --compare_base \
    --output_json "${SCRIPT_DIR}/eval_results.json"

echo ""
echo "Done. Results in ${SCRIPT_DIR}/eval_results.json"
