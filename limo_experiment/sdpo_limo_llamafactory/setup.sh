#!/usr/bin/env bash
# One-shot setup for the SDPO + LIMO LoRA pipeline.
# Builds the training venv (.venv-train) and optionally the evaluation venv (.venv-eval).
#
# Usage:
#   ./setup.sh                       # training env only (no flash-attn, no eval env)
#   ./setup.sh --with-eval           # also build .venv-eval (vLLM)
#   ./setup.sh --with-flash-attn     # also compile flash-attn 2 (10-20 min)
#   ./setup.sh --system-deps         # also apt-get python3.11-dev + git-lfs (needs sudo)
#   ./setup.sh --all                 # all of the above
#   ./setup.sh --cuda cu118          # override CUDA wheel tag (default: cu121)
#
# Idempotent: re-running reuses existing venvs and re-validates them.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WITH_EVAL=0
WITH_FLASH_ATTN=0
SYSTEM_DEPS=0
CUDA_TAG="${CUDA_TAG:-cu121}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-eval)        WITH_EVAL=1 ;;
    --with-flash-attn)  WITH_FLASH_ATTN=1 ;;
    --system-deps)      SYSTEM_DEPS=1 ;;
    --all)              WITH_EVAL=1; WITH_FLASH_ATTN=1; SYSTEM_DEPS=1 ;;
    --cuda)             shift; CUDA_TAG="$1" ;;
    -h|--help)          sed -n '1,15p' "$0"; exit 0 ;;
    *)                  echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

# ── Optional system deps (sudo) — runs first so a fresh box can self-bootstrap ──
if [[ "$SYSTEM_DEPS" == "1" ]]; then
  echo "[setup] Installing system dependencies (sudo)"
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev git git-lfs
  git lfs install
  if [[ -d "$HERE/.git" ]] || git -C "$HERE" rev-parse --git-dir >/dev/null 2>&1; then
    git lfs pull
  fi
  if [[ -d "$HOME/.config" ]] && [[ ! -w "$HOME/.config" ]]; then
    sudo chown -R "$USER:$USER" "$HOME/.config"
  fi
  echo ""
fi

# ── Pre-flight ────────────────────────────────────────────────────────────────
echo "[setup] Pre-flight checks"
command -v python3.11 >/dev/null || {
  echo "ERROR: python3.11 not found." >&2
  echo "  Install with: sudo apt install python3.11 python3.11-venv python3.11-dev" >&2
  echo "  Or re-run this script with --system-deps to install everything." >&2
  exit 1
}
# python3.11-venv is a separate apt package — `python3.11 -m venv` fails without
# it with a confusing 'ensurepip is not available' message. Catch it here.
if ! python3.11 -c "import ensurepip" 2>/dev/null; then
  echo "ERROR: python3.11-venv is not installed (ensurepip module missing)." >&2
  echo "  Install with: sudo apt install python3.11-venv" >&2
  echo "  Or re-run this script with --system-deps." >&2
  exit 1
fi
command -v git >/dev/null || { echo "ERROR: git not found" >&2; exit 1; }

if command -v nvidia-smi >/dev/null; then
  echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "  Driver CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi
echo "  CUDA wheel tag: $CUDA_TAG  (override with --cuda cuXXX)"

# ── Training env: .venv-train ─────────────────────────────────────────────────
echo ""
echo "[setup] Building .venv-train"
if [[ ! -d .venv-train ]]; then
  python3.11 -m venv .venv-train
fi
# shellcheck disable=SC1091
source .venv-train/bin/activate
pip install --upgrade pip --quiet

echo "  Installing PyTorch ($CUDA_TAG)"
pip install --quiet torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

echo "  Cloning + installing LlamaFactory (editable)"
if [[ ! -d LlamaFactory ]]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LlamaFactory
fi
pip install --quiet -e "./LlamaFactory[torch,metrics]"

if [[ "$WITH_FLASH_ATTN" == "1" ]]; then
  echo "  Compiling flash-attn (this can take 10-20 min)"
  pip install flash-attn --no-build-isolation
else
  echo "  Skipping flash-attn (training will use SDPA via flash_attn: sdpa in YAML)"
fi

echo "  Installing requirements.txt"
pip install --quiet -r requirements.txt

echo "  Verifying"
python -c "import transformers, peft, datasets, matplotlib, scipy" \
  && echo "  training imports OK"
llamafactory-cli version >/dev/null && echo "  llamafactory-cli OK"

deactivate

# ── Evaluation env: .venv-eval ────────────────────────────────────────────────
if [[ "$WITH_EVAL" == "1" ]]; then
  echo ""
  echo "[setup] Building .venv-eval"
  if [[ ! -f /usr/include/python3.11/Python.h ]]; then
    echo "  WARNING: Python.h missing — Triton compile will fail at vLLM runtime." >&2
    echo "           Re-run with --system-deps, or: sudo apt install python3.11-dev" >&2
  fi
  if [[ ! -d .venv-eval ]]; then
    python3.11 -m venv .venv-eval
  fi
  # shellcheck disable=SC1091
  source .venv-eval/bin/activate
  pip install --upgrade pip --quiet
  pip install --quiet -r requirements_eval.txt
  python -c "from vllm import LLM; from peft import PeftModel" \
    && echo "  eval imports OK"
  deactivate
fi

# ── Done ──────────────────────────────────────────────────────────────────────
cat <<EOF

[setup] Complete.

Next steps:
  1. (Optional) wandb login            # for live probe curves
  2. source .venv-train/bin/activate
  3. tmux new -s train                 # so SSH drops don't kill training
  4. WANDB_PROJECT=epistemic-recovery SKIP_PROBE=1 ./train.sh
EOF
