#!/usr/bin/env bash
# Makes the side-by-side comparison video for the frame-loss demo (runs on the VM, no board needed):
#   left   "Without VFI"  the trailer with bursts of frames dropped, each dropped frame replaced by the previous one
#   right  "With VFI"     the same drops, recovered by relay.py's VFI (the model on the CPU)
# Both sides use the same probability, max run and seed, so they drop exactly the same frames.
#
# Usage: ./make_burst_demo.sh [-p PROBABILITY] [-n MAX_RUN] [-s SEED] [-i INPUT] [-t TECHNIQUE] [-d]
#   -p  chance that a burst of lost frames starts at each frame, 0 to 1     (default 0.3)
#   -n  longest burst of consecutive lost frames                            (default 3)
#   -s  random seed                                                         (default 42)
#   -i  input video in ../videos, H.264 MP4                                 (default sintel_trailer-480p.mp4)
#   -t  VFI technique: toy_unet_tflite | linear_blend | frame_hold          (default toy_unet_tflite)
#   -d  also copy the baseline to ../videos/stuttering_burst.mp4, which demo_viewer.py's "burst" mode plays
# Output, in ../videos: stuttering_<tag>.mp4, recovered_<tag>.mp4, compare_<tag>.mp4, with tag e.g. p0.3_n3_s42.
set -euo pipefail
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2  # TensorFlow's start-up messages hide the script's own output

P=0.3 N=3 SEED=42 INPUT=sintel_trailer-480p.mp4 TECHNIQUE=toy_unet_tflite VIEWER=0
while getopts "p:n:s:i:t:dh" opt; do
	case "$opt" in
		p) P=$OPTARG ;; n) N=$OPTARG ;; s) SEED=$OPTARG ;; i) INPUT=$OPTARG ;; t) TECHNIQUE=$OPTARG ;; d) VIEWER=1 ;;
		*) sed -n '2,14p' "$0"; exit 1 ;;
	esac
done

V=../videos
TAG="p${P}_n${N}_s${SEED}"
[[ -f "$V/$INPUT" ]] || { echo "No such video: $V/$INPUT" >&2; exit 1; }

echo "== 1/3 relay: drop and recover ($TECHNIQUE, p=$P, max run $N, seed $SEED)"
python3 relay.py "$TECHNIQUE" --inference cpu --input "$INPUT" --output file --output-file "$V/recovered_$TAG.mp4" \
	--decoder cpu --scaler cpu --encoder x264 \
	--drop-probability "$P" --max-drop-run "$N" --seed "$SEED" | grep -E 'Dropped frames|Burst lengths|Recovered frames'

echo "== 2/3 baseline: the same drops, frame held"
python3 make_stuttering_baseline.py --input "$V/$INPUT" --output "$V/stuttering_$TAG.mp4" \
	--drop-probability "$P" --max-run "$N" --seed "$SEED" | grep -E 'Dropped frames|Burst lengths'

echo "== 3/3 side by side"
ffmpeg -y -loglevel error -i "$V/stuttering_$TAG.mp4" -i "$V/recovered_$TAG.mp4" -filter_complex \
	"[0:v]drawtext=text='Without VFI':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[a];\
[1:v]drawtext=text='With VFI':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[b];\
[a][b]hstack" -c:v libx264 -crf 18 "$V/compare_$TAG.mp4"

if [[ $VIEWER == 1 ]]; then
	cp "$V/stuttering_$TAG.mp4" "$V/stuttering_burst.mp4"
	echo "Copied the baseline to $V/stuttering_burst.mp4 for demo_viewer.py --mode burst"
	echo "On the board: ./run_relay.sh --drop-probability $P --max-drop-run $N --seed $SEED"
fi
echo "Done: $V/compare_$TAG.mp4"
