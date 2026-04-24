#!/usr/bin/env bash
# Watch a process's RSS and exit if it exceeds a threshold.
# Usage: watch_memory.sh <pid> <max_gb>

set -euo pipefail

PID=${1:?Usage: watch_memory.sh <pid> <max_gb>}
MAX_GB=${2:-100}
MAX_KB=$((MAX_GB * 1024 * 1024))

echo "Watching PID $PID, will alert if RSS exceeds ${MAX_GB}GB"

while true; do
    # Check if process still exists
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Process $PID no longer running. Exiting cleanly."
        exit 0
    fi

    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    if [[ -z "$RSS_KB" ]]; then
        echo "Process $PID no longer running. Exiting cleanly."
        exit 0
    fi

    RSS_GB=$(echo "scale=1; $RSS_KB / 1024 / 1024" | bc)

    if [[ "$RSS_KB" -gt "$MAX_KB" ]]; then
        echo "ALERT: PID $PID RSS is ${RSS_GB}GB (exceeds ${MAX_GB}GB limit)"
        exit 1
    fi

    sleep 60
done
