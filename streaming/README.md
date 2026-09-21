# Streaming setup

VFI (video frame interpolation) demo pipeline over the network, in two topologies.

## 2-device: Pi ↔ VM

Pi sends raw video; VM decodes, runs VFI, and writes the result to a file.

```
# On the Pi:
python scripts/two_device/sender.py --host 0.0.0.0 --port 5000

# On the VM:
python scripts/two_device/receiver.py --host <pi-ip> --port 5000 toy_unet_onnx
```

## 3-device: Pi → iMX8 → VM

Pi sends raw video; iMX8 decodes, runs VFI (NPU-accelerated), and re-encodes onward; VM just
remuxes the already-interpolated stream to a file (no VFI, no decode, no re-encode).

```
# On the Pi:
python scripts/three_device/sender.py --host 0.0.0.0 --port 5000

# On the iMX8:
python scripts/three_device/relay.py --host <pi-ip> --port 5000 \
    --listen-host 0.0.0.0 --listen-port 5001 toy_unet_onnx

# On the VM:
python scripts/three_device/receiver.py --host <imx8-ip> --port 5001
```

`sender.py` is identical in both folders (the Pi's role never changes) — duplicated rather
than shared so each folder is self-contained; keep both copies in sync if either changes.

Output video lands in `videos/received_<technique>.mp4` (2-device) or `videos/received_3hop.mp4`
(3-device) by default — override with `--output <path>`.

## Other scripts

- `scripts/standalone_inference.py` — bare model+NPU test, no GStreamer/network involved. See
  its `--help` for usage.

## Known issues

- `sender.py`'s `mpegtsmux`+`tcpserversink` chain reliably truncates ~5% of frames near the end
  of a stream (deterministic, not timing-related — confirmed across multiple property tweaks).
  Root cause not yet found; not blocking, but worth knowing before trusting output completeness.
