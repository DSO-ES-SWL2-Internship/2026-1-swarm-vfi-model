# Streaming

The scripts in this folder handle the Video Frame Interpolation (VFI) video streaming application across the three devices. 

VFI (video frame interpolation) over the network, across three devices: the Pi sends a video, the iMX8 relay
decodes it, runs VFI on every pair of frames on the NPU and re-encodes the result, and the VM shows or saves it.

```
Pi (sender.py)  # Sends the source video
    |
    |  # TCP, H.264 in MPEG-TS
    V
iMX8 (relay.py)  # Decodes the video, runs VFI over gaps of frames, re-encodes
    |            # the resulting video
    |
    |  # TCP, H.264 in MPEG-TS
    V
VM (demo_viewer.py)  # Receives the video and shows side-by-side comparison
```

Videos live in `videos/` (not tracked by git; copy them to each device).

## Running the demo

The demo shows what the VFI buys: the Pi sends the 24 fps trailer; the relay drops random bursts of 1 to 3 frames after
decoding (probability 0.3, seed 42) and recovers each lost frame with the U-Net on the NPU.

### Brief of key demo scripts

`run_sender.sh` sends the video stream stored in the Raspberry Pi's storage. 
* The script waits for the relay (i.e. i.MX8) to connect to it.
* Subsequently, sends out the video stream via TCP to the relay node.

`run_relay.sh` receives the video stream, runs VFI and sends the output.
* The script waits for the receiver (i.e. the VM running `demo_viewer.py`) to connect to it.
* It then attempts to connect to the sender.
* Subsequently, it receives the frames. It decodes the video and probabilistically drops some of the decoded frames (for the purposes of demonstration -- rather than realism or relevance to actual network-based losses).
* These dropped frames are then identified by the script, and VFI is run to fill up the gaps caused by these dropped frames, thereby restoring the video back to the original 24 fps.
* This generated 24 fps video is sent over to the VM via TCP.

`run_demo2.sh`

`demo_viewer.py` runs on the VM and plays multiple video streams side by side:
* One video stream is the video with the dropped frames `videos/stuttering_burst.mp4` (made by
`scripts/make_burst_demo.sh -d`; it must use the relay's drop settings).
* Another video is the video with the frames recovered through VFI. 
* Optionally, the original video without any dropped frames can be shown as well.
Recovered frames next to the same video with the frames dropped and held, and optionally the original.

### Executing the scripts

Start them in this order (the relay waits for the VM before connecting to the Pi). The sender and the relay exit
when the video ends, so restart both before each run.

```
# Pi
scripts/run_sender.sh            # sends videos/sintel_trailer-480p.mp4 by default

# iMX8
scripts/run_relay.sh             # PI_HOST=<ip> for the Pi's address; DROP_PROBABILITY / MAX_DROP_RUN for the losses

# VM: press Connect (or pass --connect)
python3 scripts/demo_viewer.py   # --host <iMX8's ip>
```

For the hardware acceleration demo, run `scripts/run_demo2.sh cpu` (everything on the CPU) or
`scripts/run_demo2.sh hw` (VPU decode, GPU scale, NPU, VPU encode) on the iMX8 instead of `run_relay.sh`.

## scripts/

| Script | Runs on | What it does |
| --- | --- | --- |
| `sender.py`, `run_sender.sh` | Pi | Waits for the relay to connect, then streams a video file |
| `relay.py`, `run_relay.sh` | iMX8 | Decode, scale to 448x256 grey, VFI, re-encode, serve to the VM; `--input FILE --output fake` runs it from a local file, flat out |
| `run_demo2.sh cpu\|hw` | iMX8 | The relay all on the CPU, or on the accelerators |
| `frame_dropper.py` | iMX8, VM | The random bursts of lost frames, and how a gap is filled; shared by the relay and the baseline video |
| `hw_monitor.py` | iMX8 | Live CPU / GPU / NPU / VPU load, beside any pipeline (run it in a second window during the demo) |
| `demo_viewer.py` | VM | Two-panel window (a third optional): frames dropped, recovered live, original |
| `make_burst_demo.sh`, `make_stuttering_baseline.py` | VM | The "frames dropped" baseline video, and a side-by-side comparison, for given drop settings |

`tests/test_frame_dropper.py` checks the dropper (run it with `python3 tests/test_frame_dropper.py`).

## experiments/

The pipeline benchmarks behind the optimisation results (offloading to the VPU, GPU and NPU), mostly run on the iMX8;
see `experiments/README.md`.
