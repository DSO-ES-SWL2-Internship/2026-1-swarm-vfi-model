cd ~/projects/intern1-swarm/machine-learning/streaming/scripts/test
timeout 5 gst-launch-1.0 -v filesrc location=../../videos/sintel_trailer-480p.mp4 ! qtdemux ! h264parse \
  ! vpudec frame-drop=false ! videoconvert ! videoscale ! video/x-raw,width=448,height=256,format=GRAY8 ! fakesink \
  2>&1 | grep -o "caps = video/x-raw[^\"]*" | sort -u
