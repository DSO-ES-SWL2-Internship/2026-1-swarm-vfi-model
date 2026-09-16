#!/user/bin/env bash

# receiver connects to the Pi over its AP, 60s idle-timeout as a disconnect fallback
python receiver.py --host 172.20.10.2 --port 5000 --idle-timeout 60 toy_unet
