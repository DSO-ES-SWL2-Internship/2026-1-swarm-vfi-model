#!/usr/bin/env bash
# Runs on the sender (Pi, or the iMX8 when testing iMX8 -> VM). Listens for the receiver to connect.
# Usage: [LOOP=1] ./run_sender.sh [extra sender.py args]
#   LOOP=1 restarts the sender after every stream, so benchmark.py can do repeated runs unattended.
#   Stop it with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-5000}"

if [ "${LOOP:-0}" = "1" ]; then
	while true; do
		python3 sender.py --port "$PORT" "$@"
		sleep 1
	done
else
	exec python3 sender.py --port "$PORT" "$@"
fi
