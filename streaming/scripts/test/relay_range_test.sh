#!/usr/bin/env bash
# How much CPU does the relay need per frame, for each way of handling the grey range, and without a scaler?
#   480p, G2D chain (VPU decode -> G2D scale -> CPU RGB -> grey):
#     rgb     full-range model; grey -> RGBx -> I420 on the CPU for the encoder (the original path)
#     lut     full-range model; numpy lookups to and from video range, plain copy into I420
#     video   video-range model (toy_unet_int8_video.tflite); one lookup per input frame (the G2D grey is
#             full range), plain copy into I420
#   256p, no scaler (the source is already the model's 448x256, so videoscale is a passthrough):
#     lut     full-range model; lookups in and out
#     video   video-range model; the decoder's luma plane goes to the model and back out as plain copies
# The input is the demo's lossy 12 fps video (make_lossy_video.sh), played in real time; the relay doubles it to
# 24 fps. Everything runs on the board itself (sender and receiver on localhost), so the network plays no part.
# The relay reports its own CPU time (its process only, not the sender's), inference time and lookup time.
#
# Run on the iMX8: ./relay_range_test.sh   (two runs per configuration, about 15 minutes)
# Needs artifacts/toy_unet_int8_video.tflite and the 12 fps videos in streaming/videos.
set -uo pipefail
cd "$(dirname "$0")/../three_device"
export PYTHONUNBUFFERED=1
OUT=../../../outputs/relay_range_test
mkdir -p "$OUT"

run() {
	local name=$1 video=$2
	shift 2
	for i in 1 2; do
		python3 sender.py --host 127.0.0.1 --port 5600 --video "$video" > /dev/null 2>&1 &
		local sender=$!
		sleep 2
		python3 relay.py toy_unet_tflite --host 127.0.0.1 --port 5600 --listen-port 5601 --idle-timeout 10 "$@" \
			> "$OUT/${name}_$i.txt" 2>&1 &
		local relay=$!
		# Receiver in place of the VM: retries until the relay listens (after the model's warm-up), ends with the stream.
		until gst-launch-1.0 -q tcpclientsrc host=127.0.0.1 port=5601 ! fakesink sync=false 2> /dev/null; do
			kill -0 $relay 2> /dev/null || break
			sleep 1
		done
		wait $relay
		kill $sender 2> /dev/null
		wait $sender 2> /dev/null
		echo "$name run $i: $(grep -h 'Relay CPU time' "$OUT/${name}_$i.txt")"
	done
}

run 480p_g2d_rgb sintel_trailer-480p-12fps.mp4 --scaler g2d --range rgb
run 480p_g2d_lut sintel_trailer-480p-12fps.mp4 --scaler g2d --range lut
run 480p_g2d_video sintel_trailer-480p-12fps.mp4 --scaler g2d --range video
run 256p_noscale_lut sintel_trailer-256p-12fps.mp4 --scaler cpu --range lut
run 256p_noscale_video sintel_trailer-256p-12fps.mp4 --scaler cpu --range video

echo
for f in "$OUT"/*.txt; do
	echo "== $(basename "$f" .txt)"
	grep -h -e "Relay CPU time" -e "Mean inference" -e "Mean range lookup" -e "first to last" -e "relayed:" "$f" | tail -5
done
