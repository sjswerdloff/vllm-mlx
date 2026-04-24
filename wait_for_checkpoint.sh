#!/usr/bin/env bash
# Wait for a specific checkpoint file to appear, then notify.
# Usage: wait_for_checkpoint.sh <checkpoint_path>
set -euo pipefail

CKPT=${1:?Usage: wait_for_checkpoint.sh <checkpoint_path>}

echo "Waiting for: $CKPT"
while [ ! -f "$CKPT" ]; do
    sleep 60
done
echo "Checkpoint found: $CKPT"
ls -lh "$CKPT"
