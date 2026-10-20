#!/usr/bin/env bash
# Hardware acceleration demo (runs on the iMX8, over SSH): the same relay twice, all on the CPU and then on the
# accelerators, while the video plays on the VM in demo_viewer.py. Watch the load in a second SSH window with top
# (press 1 for one row per core).
#   ./run_demo2.sh cpu   everything on the CPU: decode, scale, colour conversion, the model (4 threads), encode (MPEG-4).
#                        The CPU cannot keep up, so the video on the VM lags and all four cores are busy.
#   ./run_demo2.sh hw    the default path: VPU decode, G2D scale, model on the NPU, VPU encode. Smooth, at about
#                        80% of one core.
# For each run: start this, then the sender on the Pi (run_sender.sh), then Connect in demo_viewer.py on the VM.
# The relay stops when the video ends and prints its CPU time and inference latency. Output: /tmp/demo2_<mode>.log.
#
# Not under nnshark: its live tracer cuts the relay's TCP stream to the VM after the first frame (tested 19 Oct 2026).
# For nnshark's view of the accelerators (per-element time, NPU load), run ../test/nnshark_demo.sh --npu separately:
# the same hardware path (VPU decode, G2D scale, VPU encode, the model on the NPU), without the network.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

case "${1:-}" in
	cpu) ARGS=(--decoder cpu --scaler cpu --encoder mpeg4 --inference cpu) ;;
	hw) ARGS=(--inference npu) ;;
	*) echo "usage: $0 cpu|hw" >&2; exit 1 ;;
esac
MODE=$1
LOG=/tmp/demo2_$MODE.log

echo "Relay ($MODE): now start the sender on the Pi, then press Connect in demo_viewer.py on the VM."
python3 relay.py toy_unet_tflite --host "${PI_HOST:-172.20.10.2}" --idle-timeout "${IDLE_TIMEOUT:-60}" "${ARGS[@]}" 2>&1 |
	tee "$LOG"
