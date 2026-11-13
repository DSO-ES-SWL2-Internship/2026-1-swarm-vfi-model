# VFI on the i.MX 8M Plus

Video frame interpolation (VFI) as a way to use the i.MX 8M Plus's hardware: a small U-Net runs on the NPU, the VPU
decodes and encodes, the GPU scales. The Pi streams a video, the iMX8 relay recovers lost frames with the model, and the
VM shows the result. The question behind it: which unit is best for which job, with measurements.

```
Pi (sender) --H.264/TCP--> iMX8 relay (VPU decode -> GPU scale -> U-Net on the NPU -> VPU encode) --H.264/TCP--> VM (viewer)
```

| Folder | What is in it |
| --- | --- |
| `scripts/` | The model: architecture, training, INT8 export for the NPU, evaluation; `scripts/experiments/` the runs behind the choices; `scripts/cpp/` a C++ inference test. See `scripts/README.md` |
| `streaming/` | The three-device demo: sender, relay, viewer, and the frame-loss simulation; `streaming/experiments/` the pipeline benchmarks. See `streaming/README.md` |
| `artifacts/` | The deployed model: `vfi_unet.keras` (trained) and `vfi_unet.tflite` (INT8, what the relay loads). Everything else in it is generated |
| `splits/` | The Vimeo-90K train / val / eval triplet lists |
| `datasets/`, `outputs/` | Training data and measurement results (not tracked) |

## The model

A small U-Net (about 70 thousand parameters) whose output is the average of the two input frames plus a learned
correction. Trained on open Blender films (more motion) mixed with Vimeo-90K triplets. On held-out clips of the Sintel
trailer it scores 36.3 dB PSNR and 0.072 LPIPS, against 35.6 dB / 0.063 for a plain blend, and runs at 12.1 ms per
frame on the NPU (budget for 24 fps: 41.7 ms). The INT8 export is per-tensor (1.9x faster on the NPU than per-channel).

## Setup (on the VM)

```
python3 -m venv ~/vfi-env && source ~/vfi-env/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg
```

Data: Vimeo-90K triplets in `datasets/vimeo90k/`; the three Blender films in `datasets/blender/` (links and steps in
`scripts/README.md`); the Sintel trailer in `streaming/videos/` (the test and demo video, 480p).

```
cd scripts
python3 blender_triplets.py       # once: decodes the films and picks training windows
python3 train.py                  # -> ../artifacts/vfi_unet.keras (about 20 min on a CPU)
python3 export_tflite.py          # -> ../artifacts/vfi_unet.tflite
python3 evaluate.py               # PSNR / SSIM / LPIPS against frame hold and linear blend
```

## Running the demo

See `streaming/README.md`. On the iMX8 the relay needs, in the same layout: `streaming/scripts/` (relay.py,
frame_dropper.py, run_*.sh, hw_monitor.py) and `artifacts/vfi_unet.tflite`; the Pi needs `sender.py`, `run_sender.sh`
and `streaming/videos/sintel_trailer-480p.mp4`; the VM needs `demo_viewer.py` and the two videos. The NPU compiles the
model on its first run (about 28 s); the relay caches it under `artifacts/vx_cache/`, so run it once before presenting.

## Credits

Training data: Big Buck Bunny, Elephants Dream and Tears of Steel, (c) Blender Foundation, Creative Commons Attribution
(Big Buck Bunny and Tears of Steel: CC BY 3.0); Vimeo-90K (Xue et al., IJCV 2019). Test and demo video: the Sintel
trailer, (c) Blender Foundation | durian.blender.org, CC BY 3.0. The Sintel trailer and the films are not in this repo.
