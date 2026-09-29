#!/usr/bin/env bash
# Codec transcode benchmark (not the VFI pipeline): decode -> encode at the SAME resolution, no scaling, to see
# how CPU vs VPU decode/encode cost changes with frame size. Runs flat out over each source video;
# encoder output goes to fakesink so only decode/encode is timed. Run on the iMX8 only.
# Needs ../../videos/sintel_trailer-{480p,720p,1080p}.mp4 (missing ones are skipped).
# Usage: ./test_codec_resolution.sh [filter]   e.g. ./test_codec_resolution.sh 720p
#
# Caveat: the CPU encoder is MPEG-4 Part 2 (avenc_mpeg4) -- the board has no CPU H.264 encoder --
# so encode rows compare VPU H.264 against a cheaper CPU codec. Decode rows are like-for-like H.264.
set -euo pipefail
cd "$(dirname "$0")"

FRAMES=1251
declare -A KBPS=([480p]=2000 [720p]=4500 [1080p]=10000)  # scaled by pixel count, same for both encoders

declare -a NAMES PIPES
add() { NAMES+=("$1"); PIPES+=("$2"); }

for res in 480p 720p 1080p; do
	video="../../videos/sintel_trailer-$res.mp4"
	[[ -f "$video" ]] || { echo "skipping $res: $video not found" >&2; continue; }
	src="filesrc location=$video ! qtdemux ! h264parse"
	vpuenc="vpuenc_h264 bitrate=${KBPS[$res]}"
	cpuenc="avenc_mpeg4 bitrate=$(( KBPS[$res] * 1000 ))"
	# videoconvert passes frames straight through when formats already match, so it only costs
	# anything where a conversion is genuinely needed (e.g. vpudec's NV12 -> avenc_mpeg4).
	add "$res decode: cpu"               "$src ! avdec_h264 ! fakesink"
	add "$res decode: vpu"               "$src ! vpudec frame-drop=false ! fakesink"
	add "$res transcode: cpu -> cpu"     "$src ! avdec_h264 ! videoconvert ! $cpuenc ! fakesink"
	add "$res transcode: vpu -> vpu"     "$src ! vpudec frame-drop=false ! videoconvert ! $vpuenc ! fakesink"
	add "$res transcode: cpu -> vpu"     "$src ! avdec_h264 ! videoconvert ! $vpuenc ! fakesink"
	add "$res transcode: vpu -> cpu"     "$src ! vpudec frame-drop=false ! videoconvert ! $cpuenc ! fakesink"
done

TIMEFORMAT="%R %U %S"
FILTER="${1:-}"
printf "%-30s %8s %8s %8s\n" "variant" "wall s" "fps" "cpu s"
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
