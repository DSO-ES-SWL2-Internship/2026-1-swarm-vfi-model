#!/usr/bin/env bash
# Checks the iMX8's VPU decode/encode path in isolation (no network, no Python): decode ->
# scale to 448x256 grey (as the receiver does) -> encode -> MP4. Run on the iMX8 only.
set -euo pipefail
cd "$(dirname "$0")"

gst-launch-1.0 -e filesrc location=../../videos/sintel_trailer-480p.mp4 \
                ! qtdemux \
                ! h264parse \
                ! vpudec frame-drop=false \
                ! videoconvert \
                ! videoscale \
                ! video/x-raw, width=448, height=256, format=GRAY8 \
                ! videoconvert \
                ! vpuenc_h264 bitrate=2000 \
                ! h264parse \
                ! mp4mux \
                ! filesink location=/tmp/vpu_test.mp4

ls -lh /tmp/vpu_test.mp4
