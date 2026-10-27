# Streaming

VFI (video frame interpolation) over the network, across three devices: the Pi sends a video, the iMX8 relay
decodes it, runs VFI on every pair of frames on the NPU and re-encodes the result, and the VM shows or saves it.

```
Pi (sender.py) --TCP, H.264 in MPEG-TS--> iMX8 (relay.py) --TCP, H.264 in MPEG-TS--> VM (demo_viewer.py or receiver.py)
```

Videos live in `videos/` (not tracked by git; copy them to each device).

## Running the demo

The Pi sends the trailer with every other frame dropped (12 fps, made by `scripts/make_lossy_video.sh`); the relay
doubles it back to 24 fps; `demo_viewer.py` shows the lossy, original and recovered videos side by side.

Start them in this order (the relay waits for the VM before connecting to the Pi). The sender and the relay exit
when the video ends, so restart both before each run.

```
# Pi
scripts/run_sender.sh            # sends videos/sintel_trailer-480p-12fps.mp4 by default
# iMX8
scripts/run_relay.sh             # PI_HOST=<ip> to override the Pi's address
# VM: press Connect
python3 scripts/demo_viewer.py   # --host <iMX8's ip>
```

For the hardware acceleration demo, run `scripts/run_demo2.sh cpu` (everything on the CPU) or
`scripts/run_demo2.sh hw` (VPU decode, GPU scale, NPU, VPU encode) on the iMX8 instead of `run_relay.sh`.
To save the relayed stream to a file instead of viewing it, run `scripts/run_receiver.sh` on the VM.

## scripts/

| Script | Runs on | What it does |
| --- | --- | --- |
| `sender.py`, `run_sender.sh` | Pi | Waits for the relay to connect, then streams a video file |
| `relay.py`, `run_relay.sh` | iMX8 | Decode, scale to 448x256 grey, VFI, re-encode, serve to the VM; `--input FILE --output fake` runs it from a local file, flat out |
| `run_demo2.sh cpu\|hw` | iMX8 | The relay all on the CPU, or on the accelerators |
| `demo_viewer.py` | VM | Three-panel window: lossy 12 fps, original 24 fps, recovered 24 fps |
| `receiver.py`, `run_receiver.sh` | VM | Saves the relayed stream to `videos/received_3hop.mp4` |
| `make_lossy_video.sh` | VM | Makes the 12 fps videos (needs x264enc) |

## scripts/test/ (benchmarks and checks, mostly on the iMX8)

| Script | What it does |
| --- | --- |
| `relay_bench.py` | The relay with the model, one offload at a time, without and with queues -> `outputs/relay_bench/` |
| `pipeline_bench.py` | decode -> scale -> encode on CPU / VPU / G2D, without and with queues, with per-step predictions |
| `ocl_check.py` | GPU (OpenCL) scaler vs G2D and CPU: speed and correctness |
| `codec_suite.py`, `plot_codec_suite.py`, `plot_utilisation.py` | Codec and scaler combinations at each resolution: speed, CPU time, utilisation (plots on the VM) |
| `trace_buffers.py`, `scaler_tlb.py` | Frame memory per hand-over between units; why the CPU scaler is slower on VPU frames |
| `test_g2d_colour.sh` | Which colorimetry setting gives the right grey on the G2D path |
| `hw_monitor.py` | Live CPU / GPU / NPU / VPU load, beside any pipeline |
| `nnshark_demo.sh` | The relay's video path under NXP's nnshark live profiler |
| `compare_videos.sh` | Plays an original and a received video side by side |
