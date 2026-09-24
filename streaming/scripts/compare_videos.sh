#!/usr/bin/env bash
# Plays the original and a received video side by side, scaled up, for visual comparison.
# Usage: ./compare_videos.sh <received.mp4> [original.mp4]
#   PANEL_WIDTH=1200 ./compare_videos.sh ...   larger panels (height follows the 448x256 aspect ratio)
set -euo pipefail
cd "$(dirname "$0")"

RECEIVED="${1:?usage: $0 <received.mp4> [original.mp4]}"
ORIGINAL="${2:-../videos/sintel_trailer-480p.mp4}"
PANEL_WIDTH="${PANEL_WIDTH:-896}"
PANEL_HEIGHT=$(( PANEL_WIDTH * 256 / 448 ))
SINK="${SINK:-ximagesink}"
LABEL="$(basename "$RECEIVED" .mp4)"

panel() {  # $1 = file, $2 = demuxer name, $3 = label
	echo "filesrc location=\"$1\" ! qtdemux name=$2 $2.video_0 ! queue ! decodebin ! videoconvert ! videoscale \
	! video/x-raw,width=$PANEL_WIDTH,height=$PANEL_HEIGHT,pixel-aspect-ratio=1/1 \
	! textoverlay text=\"$3\" valignment=top halignment=left font-desc=\"Sans 20\""
}

eval gst-launch-1.0 \
	compositor name=comp sink_1::xpos="$PANEL_WIDTH" ! videoconvert ! "$SINK" \
	"$(panel "$ORIGINAL" d0 "Original")" ! comp.sink_0 \
	"$(panel "$RECEIVED" d1 "Received: $LABEL")" ! comp.sink_1
