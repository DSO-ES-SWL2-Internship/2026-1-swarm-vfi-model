# Set up the three devices

The demo uses three machines on the same network. Set each one up once. This page tells you what goes on each device,
and why.

| Device | Role | What runs there | Needs |
| --- | --- | --- | --- |
| **VM** (your computer) | Develop, train, view | Model scripts, `demo_viewer.py` | Python 3.10, GStreamer, GTK, ffmpeg |
| **Raspberry Pi** | The "camera" | `sender.py` | GStreamer, the video file |
| **i.MX8 board** | The relay | `relay.py`, `hw_monitor.py` | The Yocto image, the model |

!!! info "Addresses used in this guide"
    The scripts default to a phone-hotspot network: **Pi `172.20.10.2`**, **board `172.20.10.10`**. If yours differ,
    set `PI_HOST=<ip>` when you start the relay, and pass `--host <ip>` to the viewer.

## 1. The VM

### Python environment

```bash
git clone https://github.com/DSO-ES-SWL2-Internship/2026-1-swarm-vfi-model.git
cd 2026-1-swarm-vfi-model
python3 -m venv ~/vfi-env && source ~/vfi-env/bin/activate
pip install -r requirements.txt
```

This installs TensorFlow (training and export), PyTorch and LPIPS (only for the perceptual score) and a few helpers.

!!! warning "TensorFlow and PyTorch in one process"
    If a script imports both, import `torch` **first**. The other way round crashes. `model/evaluate.py` already does
    this; you only need to care if you write your own.

### System packages

```bash
sudo apt install ffmpeg python3-gi gir1.2-gtk-3.0 \
    gstreamer1.0-tools gstreamer1.0-gtk3 gstreamer1.0-libav \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

`ffmpeg` decodes video for training and evaluation. The rest runs the viewer window (GTK) and its GStreamer pipelines.
Run the viewer with the system `python3`, which has `gi` (PyGObject).

### Data (only if you want to retrain)

You do not need any data to run the demo. To train the model you need:

- **Vimeo-90K triplets** in `datasets/vimeo90k/`
- **Three Blender films** in `datasets/blender/{bbb,ed,tos}/`, from download.blender.org
- **The Sintel trailer** (480p) at `streaming/videos/sintel_trailer-480p.mp4`, which is also the demo video

The exact links and the steps to turn the films into training windows are in [The model](ml-model.md#train-it).

## 2. The Raspberry Pi

The Pi only streams a video file, so it needs very little.

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

Copy three things from the VM, keeping this layout (the sender finds the video relative to its own location):

```
streaming/scripts/sender.py
streaming/scripts/run_sender.sh
streaming/videos/sintel_trailer-480p.mp4
```

Get the Pi onto the same Wi-Fi as the board, then check from the VM: `ssh pi@172.20.10.2`.

## 3. The i.MX8 board

### 3a. The image

The board runs a custom Linux image built with Yocto. It already contains everything the relay needs: TensorFlow
Lite 2.19 with NXP's VX delegate (the NPU driver), GStreamer with the NXP plugins for the VPU and GPU, Python with
`numpy`, and Wi-Fi with auto-connect. You do not need to install any of it on the board.

- To **use** an existing SD card: set the boot switches to **0011** (SD card boot), insert the card, power on.
- To **build or change** the image: see [The Yocto image](yocto.md).

### 3b. Get on the network

The board connects to Wi-Fi by itself on boot (interface `mlan0`). Find its address on the serial console (the debug
USB port, `/dev/ttyUSB2`, 115200 baud, log in as `root`, no password), then `ssh root@<board ip>` from the VM.

If it did not connect, run `wpa_cli -i mlan0 scan` and then `udhcpc` on the serial console. See
[Tips and troubleshooting](tips.md#the-board-does-not-answer-ssh).

### 3c. Check the NPU works

Before anything else, check that a model runs on the NPU. This is the most reliable test that the delegate and the
image are healthy:

```bash
/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model \
    --graph=vfi_unet.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
```

Look for `Inference (avg)` of about 12 ms. The **first** inference takes about 28 seconds, because the NPU compiles the
model's graph. Later runs are fast.

### 3d. Copy the relay and the model

From the VM, in the repository folder:

```bash
ssh root@172.20.10.10 mkdir -p vfi/streaming/scripts vfi/artifacts
scp streaming/scripts/{relay.py,frame_dropper.py,hw_monitor.py,run_relay.sh,run_demo2.sh} \
    root@172.20.10.10:vfi/streaming/scripts/
scp artifacts/vfi_unet.tflite root@172.20.10.10:vfi/artifacts/
```

The relay finds the model at `../../artifacts/vfi_unet.tflite` relative to the script, which is why the folder layout
matters. It also keeps the compiled NPU graph in `artifacts/vx_cache/`, so the 28 seconds is paid only once.

!!! warning "Run from one copy only"
    If the board has an older clone of the repository, make sure you start the relay from `~/vfi`, not from the old
    clone. The old copy has the old relay and the old model, and it fails in confusing ways. The relay prints which
    model it loads: look for `vfi_unet.tflite`.

### 3e. Turn on the load monitor (once per boot)

`hw_monitor.py` shows GPU and NPU load from the kernel's debug filesystem:

```bash
mount -t debugfs none /sys/kernel/debug     # "already mounted" is fine
```

## Check everything

| Check | Command | Expect |
| --- | --- | --- |
| VM reaches the board | `ssh root@172.20.10.10` | a shell |
| VM reaches the Pi | `ssh pi@172.20.10.2` | a shell |
| NPU runs the model | the `benchmark_model` command above | about 12 ms |
| Relay works on its own | on the board: `cd ~/vfi/streaming/scripts && python3 relay.py toy_unet_tflite --input <trailer.mp4> --output fake` | 1251 frames in and out |

Then go to [Run the demos](run-the-demos.md).
