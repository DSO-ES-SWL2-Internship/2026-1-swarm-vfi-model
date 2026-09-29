#!/usr/bin/env bash
# G2D only outputs RGB, so VPU -> G2D -> grey goes YUV -> RGB (G2D) -> grey (videoconvert). If the two
# steps use different range/colour-matrix conventions, the grey values drift from the original's
# luma. This writes one output per colorimetry setting on the GRAY8 caps (range:matrix:transfer:primaries)
# so they can be compared against a reference on the VM. Run on the iMX8 only.
set -euo pipefail
cd "$(dirname "$0")"

SRC="filesrc location=../../videos/sintel_trailer-480p.mp4 ! qtdemux ! h264parse ! vpudec frame-drop=false"
G2D="imxvideoconvert_g2d ! video/x-raw,width=448,height=256"
ENC="videoconvert ! vpuenc_h264 bitrate=2000 ! h264parse ! mp4mux"

declare -A VARIANTS=(
	[default]="video/x-raw,format=GRAY8"
	[limited_bt601]="video/x-raw,format=GRAY8,colorimetry=2:4:0:0"
	[limited_bt709]="video/x-raw,format=GRAY8,colorimetry=2:3:0:0"
	[full_bt709]="video/x-raw,format=GRAY8,colorimetry=1:3:0:0"
)

for name in "${!VARIANTS[@]}"; do
	out="/tmp/g2d_colour_${name}.mp4"
	gst-launch-1.0 -q -e $SRC ! $G2D ! videoconvert ! ${VARIANTS[$name]} ! $ENC ! filesink location="$out" \
		&& echo "wrote $out" || echo "FAILED: $name"
done
