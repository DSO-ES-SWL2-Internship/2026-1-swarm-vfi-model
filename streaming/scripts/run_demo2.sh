#!/usr/bin/env bash
# Hardware acceleration demo (runs on the iMX8, over SSH): the frame-loss demo twice, all on the CPU and then on the
# accelerators, while the video plays on the VM in demo_viewer.py. Watch the load in a second SSH window with top
# (press 1 for one row per core), or hw_monitor.py.
#   ./run_demo2.sh cpu   everything on the CPU: decode, scale, colour conversion, the model (4 threads), encode (MPEG-4).
#                        The CPU cannot keep up (65 s for the 52 s trailer), so the video on the VM lags and all four
#                        cores are busy.
#   ./run_demo2.sh hw    the default path: VPU decode, GPU scale (OCL), model on the NPU, VPU encode.
# For each run: start this, then the sender on the Pi (run_sender.sh), then Connect in demo_viewer.py on the VM.
# The losses are the same as run_relay.sh's (DROP_PROBABILITY 0.3, MAX_DROP_RUN 3); extra options go to relay.py.
# The relay stops when the video ends and prints its CPU time and inference latency. Output: /tmp/demo2_<mode>.log.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
DROP_PROBABILITY="${DROP_PROBABILITY:-0.3}"
MAX_DROP_RUN="${MAX_DROP_RUN:-3}"

case "${1:-}" in
	cpu) ARGS=(--decoder cpu --scaler cpu --encoder mpeg4 --inference cpu) ;;
	hw) ARGS=(--inference npu) ;;
	*) echo "usage: $0 cpu|hw [relay.py options]" >&2; exit 1 ;;
esac
MODE=$1
shift  # anything after cpu|hw goes to relay.py
LOG=/tmp/demo2_$MODE.log

echo "Relay ($MODE): now start the sender on the Pi, then press Connect in demo_viewer.py on the VM."
python3 relay.py toy_unet_tflite --host "${PI_HOST:-172.20.10.2}" --idle-timeout "${IDLE_TIMEOUT:-60}" "${ARGS[@]}" \
	--drop-probability "$DROP_PROBABILITY" --max-drop-run "$MAX_DROP_RUN" "$@" 2>&1 |
	tee "$LOG"
