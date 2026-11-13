#!/usr/bin/env bash
# Runs on the iMX8. Connects to the Pi, runs VFI, and listens for the VM to connect on LISTEN_PORT.
# Usage: [PI_HOST=ip] [TECHNIQUE=name] ./run_relay.sh [extra relay.py args]
# The frame-loss demo: the Pi sends the 24 fps trailer (run_sender.sh); the relay drops random bursts of frames after
# decoding and recovers them with the U-Net on the NPU. DROP_PROBABILITY and MAX_DROP_RUN set the losses (defaults 0.3
# and 3; the baseline video in demo_viewer.py is made with the same values, see make_burst_demo.sh). It scales each frame
# to the model's 448x256 on the GPU (OCL, the default after the VPU decoder; --scaler cpu for the CPU).
set -euo pipefail
cd "$(dirname "$0")"
# Print Python's output immediately even when piped (e.g. | tee log.txt), instead of buffering it.
export PYTHONUNBUFFERED=1

PI_HOST="${PI_HOST:-172.20.10.2}"
PI_PORT="${PI_PORT:-5000}"
LISTEN_PORT="${LISTEN_PORT:-5001}"
TECHNIQUE="${TECHNIQUE:-toy_unet_tflite}"  # NPU path; passthrough | frame_hold | linear_blend also work
IDLE_TIMEOUT="${IDLE_TIMEOUT:-60}"
DROP_PROBABILITY="${DROP_PROBABILITY:-0.3}"
MAX_DROP_RUN="${MAX_DROP_RUN:-3}"

exec python3 relay.py "$TECHNIQUE" --host "$PI_HOST" --port "$PI_PORT" \
	--listen-port "$LISTEN_PORT" --idle-timeout "$IDLE_TIMEOUT" \
	--drop-probability "$DROP_PROBABILITY" --max-drop-run "$MAX_DROP_RUN" "$@"
