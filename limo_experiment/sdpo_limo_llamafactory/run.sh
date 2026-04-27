#!/usr/bin/env bash
# Convenience wrapper: runs train.sh then eval.sh with the same RUN_ID.
# For most experiments you'll want to run them separately:
#   ./train.sh         (blocks until training + merge complete)
#   ./eval.sh          (reads RUN_ID from .last_run_id automatically)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "=== train.sh (RUN_ID=$RUN_ID) ==="
bash "$HERE/train.sh"

echo ""
echo "=== eval.sh (RUN_ID=$RUN_ID) ==="
bash "$HERE/eval.sh"
