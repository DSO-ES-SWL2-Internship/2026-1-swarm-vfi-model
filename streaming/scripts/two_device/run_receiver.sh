#!/usr/bin/env bash
# Runs on the VM. Connects to the sender and saves the (optionally VFI'd) stream to streaming/videos/.
# Usage: [SENDER_HOST=ip] [TECHNIQUE=name] ./run_receiver.sh [extra receiver.py args]
#   Pi sender:   ./run_receiver.sh
#   iMX8 sender: SENDER_HOST=172.20.10.5 ./run_receiver.sh
set -euo pipefail
cd "$(dirname "$0")"

# SENDER_HOST="${SENDER_HOST:-172.20.10.2}"  # Uncomment for Pi sender
SENDER_HOST="${SENDER_HOST:-172.20.10.5}"  # Uncomment for iMX8 sender
PORT="${PORT:-5000}"
TECHNIQUE="${TECHNIQUE:-toy_unet_tflite}"     # passthrough | frame_hold | linear_blend | toy_unet_onnx | toy_unet_tflite
IDLE_TIMEOUT="${IDLE_TIMEOUT:-60}"

exec python3 receiver.py "$TECHNIQUE" --host "$SENDER_HOST" --port "$PORT" --idle-timeout "$IDLE_TIMEOUT" "$@"
