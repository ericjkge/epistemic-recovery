#!/usr/bin/env bash
# Convenience wrapper: runs train.sh then eval.sh under the same EXPERIMENT.
# For most experiments you'll want to run them separately:
#   ./train.sh         (blocks until training + merge complete)
#   ./eval.sh          (reads EXPERIMENT from .last_experiment automatically)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# If EXPERIMENT isn't set, train.sh auto-picks the next experimentN and writes
# it to .last_experiment, which eval.sh then reads.
echo "=== train.sh ==="
bash "$HERE/train.sh"

echo ""
echo "=== eval.sh ==="
bash "$HERE/eval.sh"
