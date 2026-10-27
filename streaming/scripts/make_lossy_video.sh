#!/usr/bin/env bash
# Makes the "lossy" video for the 3-panel demo: the 24 fps trailer with every other frame dropped, so 12 fps.
# The demo sends it from the Pi, the relay's VFI doubles it back to 24 fps, and demo_viewer.py shows it next
# to the original: can VFI recover the frames that were lost?
#
# videorate drop-only keeps the frame nearest each 1/12 s slot, i.e. frames 0, 2, 4, ... at their original
# timestamps, so each interpolated frame stands in for a dropped odd frame of the original. The kept frames
# are re-encoded (dropping frames breaks the H.264 references), at high quality so the re-encoding barely shows.
#
# 256p: the 480p trailer scaled to the model's own size, 448x256, at 24 fps and at 12 fps. With the source at
# model size the relay's scaler has nothing to resize: the decoder's luma plane goes straight to the model.
#
# Run on the VM (needs x264enc), then copy the output(s) to the Pi's videos folder too:
#   ./make_lossy_video.sh [480p]   -> ../videos/sintel_trailer-480p-12fps.mp4
#   ./make_lossy_video.sh 256p     -> ../videos/sintel_trailer-256p.mp4 and sintel_trailer-256p-12fps.mp4
set -euo pipefail
cd "$(dirname "$0")"

RES="${1:-480p}"
VIDEOS=../videos
ENCODE="x264enc pass=quant quantizer=16 speed-preset=medium key-int-max=12 ! video/x-h264,profile=high ! h264parse ! mp4mux"

if [ "$RES" = 256p ]; then
	gst-launch-1.0 -q -e filesrc location=$VIDEOS/sintel_trailer-480p.mp4 ! qtdemux ! h264parse ! avdec_h264 ! \
		videoscale ! video/x-raw,width=448,height=256 ! $ENCODE ! filesink location=$VIDEOS/sintel_trailer-256p.mp4
	echo "Saved $VIDEOS/sintel_trailer-256p.mp4"
fi

gst-launch-1.0 -q -e filesrc location="$VIDEOS/sintel_trailer-$RES.mp4" ! qtdemux ! h264parse ! avdec_h264 ! \
	videorate drop-only=true ! video/x-raw,framerate=12/1 ! $ENCODE ! filesink location="$VIDEOS/sintel_trailer-$RES-12fps.mp4"
echo "Saved $VIDEOS/sintel_trailer-$RES-12fps.mp4"
