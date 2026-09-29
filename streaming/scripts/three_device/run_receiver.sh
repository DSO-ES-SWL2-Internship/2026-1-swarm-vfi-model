#!/usr/bin/env bash
# Runs on the VM. Connects to the iMX8 relay and saves the final stream to streaming/videos/.
# Usage: [IMX8_HOST=ip] ./run_receiver.sh [extra receiver.py args]
set -euo pipefail
cd "$(dirname "$0")"
# Print Python's output immediately even when piped (e.g. | tee log.txt), instead of buffering it.
export PYTHONUNBUFFERED=1

IMX8_HOST="${IMX8_HOST:-172.20.10.8}"
PORT="${PORT:-5001}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-60}"

exec python3 receiver.py --host "$IMX8_HOST" --port "$PORT" --idle-timeout "$IDLE_TIMEOUT" "$@"
