#!/usr/bin/env bash
# Hardware acceleration demo (runs on the iMX8, over SSH): the relay under nnshark's live view, which shows CPU load per
# core, GPU/NPU load, DDR traffic and each GStreamer element's time per frame while the video plays on the VM.
# Two runs, one after the other:
#   ./run_demo2.sh cpu   everything on the CPU: decode, scale, colour conversion, the model (4 threads), encode (MPEG-4).
#                        The CPU cannot keep up, so the video on the VM lags and the accelerators sit idle.
#   ./run_demo2.sh hw    the default path: VPU decode, G2D scale, model on the NPU, VPU encode. Smooth, and the load
#                        moves from the CPU to the accelerators.
# For each run: start this, then the sender on the Pi (run_sender.sh), then Connect in demo_viewer.py on the VM.
# Press q to quit the live view (the relay also stops when the video ends). Relay output: /tmp/demo2_<mode>.log.
#
# GPU/NPU load needs the board booted with galcore.gpuProfiler=1 galcore.powerManagement=0 (see nnshark_demo.sh).
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
	cpu) ARGS=(--decoder cpu --scaler cpu --encoder mpeg4 --inference cpu) ;;
	hw) ARGS=(--inference npu) ;;
	*) echo "usage: $0 cpu|hw" >&2; exit 1 ;;
esac
MODE=$1
LOG=/tmp/demo2_$MODE.log

grep -q "galcore.gpuProfiler=1" /proc/cmdline ||
	echo "Note: galcore.gpuProfiler=1 is not on the kernel command line, so GPU/NPU load may not show."
pkill -9 -f nnshark-perf-ddr.sh 2>/dev/null || true  # a leftover from an earlier run inflates the DDR numbers
trap 'reset 2>/dev/null || stty sane' EXIT  # the live view can leave the terminal in raw mode

echo "Relay ($MODE) starting under nnshark; its output goes to $LOG."
echo "Now start the sender on the Pi, then press Connect in demo_viewer.py on the VM."
sleep 3
# The relay's own messages go to the log from inside Python (sys.stdout), so that the terminal itself stays free for
# nnshark's live view, which draws on it from C. A shell redirect (> log) would send the live view to the log too.
GST_DEBUG="GST_TRACER:7" GST_TRACERS="live" python3 -c '
import runpy, sys
sys.stdout = sys.stderr = open(sys.argv[1], "w", buffering=1)
sys.argv = ["relay.py", *sys.argv[2:]]
runpy.run_path("relay.py", run_name="__main__")
' "$LOG" toy_unet_tflite --host "${PI_HOST:-172.20.10.2}" --idle-timeout "${IDLE_TIMEOUT:-60}" "${ARGS[@]}"
