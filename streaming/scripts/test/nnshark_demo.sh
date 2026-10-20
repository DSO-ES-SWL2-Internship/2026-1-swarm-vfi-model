#!/usr/bin/env bash
# Live utilisation demo with nnshark (NXP's GStreamer profiler): plays the relay's video path in real time
# (VPU decode -> G2D scale to 448x256 -> VPU encode) under nnshark's "live" view, which shows CPU load per
# core, 3D GPU / NPU / G2D load, DDR traffic and each element's time per frame. Run on the iMX8, over SSH
# (the view doesn't refresh well on the serial console). Press q to quit.
# Usage: ./nnshark_demo.sh [--npu] [480p|720p|1080p]   (default 1080p)
#   --npu  also runs the U-Net on the NPU in the background (benchmark_model), so the NPU load shows too.
#
# GPU/NPU load needs the board booted with galcore.gpuProfiler=1 galcore.powerManagement=0 (NXP UG10166
# sec 9.3.1); the script warns if they are missing. With power management off, don't use this boot for
# speed or CPU-time measurements. Known gaps on this BSP: G2D load reads nan, and the per-unit DDR rows also
# count other units' traffic (the totals are fine).
set -euo pipefail
cd "$(dirname "$0")"

NPU=0
RES=1080p
for arg in "$@"; do
	case "$arg" in
		--npu) NPU=1 ;;
		480p|720p|1080p) RES=$arg ;;
		*) echo "usage: $0 [--npu] [480p|720p|1080p]" >&2; exit 1 ;;
	esac
done

VIDEO="../../videos/sintel_trailer-$RES.mp4"
MODEL="../../../artifacts/toy_unet_int8.tflite"
BENCHMARK=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model
DELEGATE=/usr/lib/libvx_delegate.so

[[ -f "$VIDEO" ]] || { echo "$VIDEO not found: copy it from the VM first" >&2; exit 1; }
[[ $NPU == 0 || -f "$MODEL" ]] || { echo "$MODEL not found: copy it from the VM first" >&2; exit 1; }
[[ -n "${SSH_CONNECTION:-}" ]] || echo "Note: not an SSH session; the live view may not render well on the serial console."
grep -q "galcore.gpuProfiler=1" /proc/cmdline ||
	echo "Note: galcore.gpuProfiler=1 is not on the kernel command line, so GPU/NPU load may not show (see header)."

# The live tracer's DDR loop (/tmp/nnshark-perf-ddr.sh, perf every second) outlives the pipeline, and a leftover one
# inflates the DDR numbers (NXP UG10166 sec 9.3.3). The board's BusyBox has no pkill, hence ps + awk.
stop_ddr_loops() { ps -o pid,args | awk '/[n]nshark-perf-ddr/ {print $1}' | xargs -r kill -9 2>/dev/null || true; }
stop_ddr_loops

BENCH_PID=""
cleanup() {
	[[ -n "$BENCH_PID" ]] && kill "$BENCH_PID" 2>/dev/null || true
	stop_ddr_loops
	reset 2>/dev/null || stty sane  # the live view can leave the terminal in raw mode if quit with Ctrl+C
}
trap cleanup EXIT

if [[ $NPU == 1 ]]; then
	# Enough runs to outlast the video (~12 ms each); killed on exit. Output to a log so it doesn't mix with the view.
	"$BENCHMARK" --graph="$MODEL" --external_delegate_path="$DELEGATE" --num_runs=100000 > /tmp/nnshark_demo_npu.log 2>&1 &
	BENCH_PID=$!
	echo "Model running on the NPU in the background (log: /tmp/nnshark_demo_npu.log); waiting for its warm-up..."
	sleep 5
fi

echo "Starting nnshark ($RES, real time, ~52 s). Arrow keys move around, q quits."
sleep 2
GST_DEBUG="GST_TRACER:7" GST_TRACERS="live" gst-launch-1.0 -q \
	filesrc location="$VIDEO" ! qtdemux ! h264parse ! vpudec frame-drop=false ! \
	imxvideoconvert_g2d ! video/x-raw,width=448,height=256 ! \
	vpuenc_h264 bitrate=2000 ! fakesink sync=true
