#!/usr/bin/env bash
# Runs on the iMX8. Connects to the Pi, runs VFI, and listens for the VM to connect on LISTEN_PORT.
# Usage: [PI_HOST=ip] [TECHNIQUE=name] ./run_relay.sh [extra relay.py args]
set -euo pipefail
cd "$(dirname "$0")"

PI_HOST="${PI_HOST:-172.20.10.2}"
PI_PORT="${PI_PORT:-5000}"
LISTEN_PORT="${LISTEN_PORT:-5001}"
TECHNIQUE="${TECHNIQUE:-toy_unet_tflite}"  # NPU path; passthrough | frame_hold | linear_blend | toy_unet_onnx also work
IDLE_TIMEOUT="${IDLE_TIMEOUT:-60}"

exec python3 relay.py "$TECHNIQUE" --host "$PI_HOST" --port "$PI_PORT" \
	--listen-port "$LISTEN_PORT" --idle-timeout "$IDLE_TIMEOUT" "$@"
