#!/usr/bin/env bash
# Runs on the sender (Pi, or the iMX8 when testing iMX8 -> VM). Listens for the receiver to connect.
# Usage: ./run_sender.sh [extra sender.py args]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-5000}"

exec python3 sender.py --port "$PORT" "$@"
