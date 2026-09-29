#!/usr/bin/env bash
# Times decode/encode pipeline variants on the iMX8 to find where the VPU path loses time.
# Each runs flat out (no pacing) over the full video. Run on the iMX8 only.
# "cpu s" = user+sys CPU time: lower means more work offloaded from the CPU.
# Usage: [VIDEO=path] ./test_codec_matrix.sh [filter]   e.g. VIDEO=../../videos/sintel_trailer-1080p.mp4 ./test_codec_matrix.sh scale
set -euo pipefail
cd "$(dirname "$0")"

VIDEO="${VIDEO:-../../videos/sintel_trailer-480p.mp4}"
FRAMES=1251
SRC="filesrc location=$VIDEO ! qtdemux ! h264parse"
SCALE="videoconvert ! videoscale ! video/x-raw,width=448,height=256,format=GRAY8 ! videoconvert"
VPUDEC="vpudec frame-drop=false"
VPUENC="vpuenc_h264 bitrate=2000 ! h264parse ! mp4mux"
CPUENC="avenc_mpeg4 bitrate=2000000 ! mpeg4videoparse ! mp4mux"

declare -a NAMES PIPES
add() { NAMES+=("$1"); PIPES+=("$2"); }

add "decode only: cpu"                "$SRC ! avdec_h264 ! fakesink"
add "decode only: vpu"                "$SRC ! $VPUDEC ! fakesink"
add "full: cpu dec + cpu enc"         "$SRC ! avdec_h264 ! $SCALE ! $CPUENC ! filesink location=/tmp/m.mp4"
add "full: vpu dec + vpu enc"         "$SRC ! $VPUDEC ! $SCALE ! $VPUENC ! filesink location=/tmp/m.mp4"
add "full: vpu, system memory"        "$SRC ! $VPUDEC use-vpu-memory=false ! $SCALE ! $VPUENC ! filesink location=/tmp/m.mp4"
add "full: vpu, queues between"       "$SRC ! $VPUDEC ! queue ! $SCALE ! queue ! $VPUENC ! filesink location=/tmp/m.mp4"
add "full: vpu dec + cpu enc"         "$SRC ! $VPUDEC ! $SCALE ! $CPUENC ! filesink location=/tmp/m.mp4"
add "full: cpu dec + vpu enc"         "$SRC ! avdec_h264 ! $SCALE ! $VPUENC ! filesink location=/tmp/m.mp4"

# Isolates the decode -> scale/grey step, and tries the 2D GPU engine (G2D) for scaling after vpudec.
G2DSCALE="imxvideoconvert_g2d ! video/x-raw,width=448,height=256 ! videoconvert ! video/x-raw,format=GRAY8"
add "scale: cpu dec + cpu scale"      "$SRC ! avdec_h264 ! $SCALE ! fakesink"
add "scale: vpu dec + cpu scale"      "$SRC ! $VPUDEC ! $SCALE ! fakesink"
add "scale: vpu dec + g2d scale"      "$SRC ! $VPUDEC ! $G2DSCALE ! fakesink"
add "full: vpu dec + g2d + vpu enc"   "$SRC ! $VPUDEC ! $G2DSCALE ! videoconvert ! $VPUENC ! filesink location=/tmp/m.mp4"
add "full: cpu dec + g2d + vpu enc"   "$SRC ! avdec_h264 ! $G2DSCALE ! videoconvert ! $VPUENC ! filesink location=/tmp/m.mp4"

TIMEFORMAT="%R %U %S"
[[ -f "$VIDEO" ]] || { echo "video not found: $VIDEO" >&2; exit 1; }
echo "video: $VIDEO"
printf "%-30s %8s %8s %8s\n" "variant" "wall s" "fps" "cpu s"
FILTER="${1:-}"  # optional: only run variants whose name contains this text, e.g. "g2d" or "scale"
for i in "${!NAMES[@]}"; do
	[[ -n "$FILTER" && "${NAMES[$i]}" != *"$FILTER"* ]] && continue
	if ! t=$( { time gst-launch-1.0 -q -e ${PIPES[$i]} >/dev/null 2>&1; } 2>&1 ); then
		printf "%-30s %s\n" "${NAMES[$i]}" "FAILED (rerun its pipeline by hand to see the error)"
		continue
	fi
	read -r real user sys <<< "$t"
	awk -v n="${NAMES[$i]}" -v r="$real" -v u="$user" -v s="$sys" -v f="$FRAMES" \
		'BEGIN { printf "%-30s %8.1f %8.1f %8.1f\n", n, r, f / r, u + s }'
done
