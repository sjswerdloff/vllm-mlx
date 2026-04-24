#!/usr/bin/env bash
# Wait for a file to gain a new line, then print the last line.
# Usage: wait_for_new_line.sh <file>
set -euo pipefail

FILE=${1:?Usage: wait_for_new_line.sh <file>}
INITIAL=$(wc -l < "$FILE")
echo "Watching $FILE (currently $INITIAL lines)"

while true; do
    CURRENT=$(wc -l < "$FILE")
    if [ "$CURRENT" -gt "$INITIAL" ]; then
        echo "New line detected ($INITIAL -> $CURRENT)"
        tail -1 "$FILE"
        exit 0
    fi
    sleep 5
done
