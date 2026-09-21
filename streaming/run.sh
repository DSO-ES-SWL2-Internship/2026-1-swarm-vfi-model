#!/user/bin/env bash

# 2-device (Pi <-> VM) setup: receiver connects to the Pi over its AP, 60s idle-timeout as a
# disconnect fallback. For the 3-device (Pi -> iMX8 -> VM) setup, run scripts/three_device/
# relay.py on the iMX8 and scripts/three_device/receiver.py here instead.
python scripts/two_device/receiver.py --host 172.20.10.2 --port 5000 --idle-timeout 60 toy_unet_onnx
