#!/usr/bin/env bash
# Runs on the Pi. Listens for the iMX8 relay to connect.
# Usage: ./run_sender.sh [extra sender.py args]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-5000}"

exec python3 sender.py --port "$PORT" "$@"
