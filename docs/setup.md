# Set up the three devices

The demo uses three machines on the same network. Set each one up once. This page tells you what goes on each device,
and why.

| Device | Role | What runs there | Needs |
| --- | --- | --- | --- |
| **Raspberry Pi** | The "camera" | `sender.py` | GStreamer, the video file |
| **i.MX8 board** | The relay | `relay.py`, `hw_monitor.py` | The Yocto image, the model |
| **VM** (your computer) | Develop, train, view | Model scripts, `demo_viewer.py` | Python 3.10, GStreamer, GTK, ffmpeg |

!!! info "Addresses used in this guide"
    The scripts default to a phone-hotspot network: **Pi `172.20.10.2`**, **board `172.20.10.10`**. If yours differ,
    set `PI_HOST=<ip>` when you start the relay, and pass `--host <ip>` to the viewer.

## The videos you need

The demo plays one video, the *Sintel* trailer. Each device needs different files, all in the repository's
`streaming/videos/` folder:

| Device | File | What it is for |
| --- | --- | --- |
| **Raspberry Pi** | `sintel_trailer-480p.mp4` | The video the Pi streams to the board |
| **i.MX8 board** | none | The relay receives the video from the Pi, so it needs no video file. (To test it without the Pi, see `--input` in [The GStreamer pipeline](gstreamer-pipeline.md#options).) |
| **VM** | `sintel_trailer-480p.mp4` | The viewer's "Original" panel. It is hidden by default, but the viewer will not start without the file |
| **VM** | `stuttering_burst.mp4` | The viewer's "Without VFI" panel: the same video with the same frames dropped and held |

**1. The trailer.** Download `sintel_trailer-480p.mp4` from
[download.blender.org/durian/trailer](https://download.blender.org/durian/trailer/) and put it in `streaming/videos/` on
the VM (the folder is not in git, so create it with `mkdir -p streaming/videos`). The Pi gets its own copy: section 2
shows how.

**2. The "without VFI" video.** Make it on the VM, from the trailer, with the same drop settings as the relay
(probability 0.3, bursts of up to 3 frames, seed 42):

```bash
cd streaming/scripts
python3 make_stuttering_baseline.py --input ../videos/sintel_trailer-480p.mp4 \
    --output ../videos/stuttering_burst.mp4 --drop-probability 0.3 --max-run 3 --seed 42
```

??? info "If Retraining the Model"
    The demo needs only the videos above. To **train** the model you also need:

    - **Vimeo-90K triplets** in `datasets/vimeo90k/`
    - **Three Blender films** in `datasets/blender/{bbb,ed,tos}/` (*Big Buck Bunny*, *Elephants Dream* and *Tears of
      Steel*), from download.blender.org

    The trailer is used for testing as well. The steps are in [The model](ml-model.md#train-it), and the download details
    are in `model/README.md`.

## 1. The VM

### Get the code

=== "With git"

    ```bash
    git clone https://github.com/DSO-ES-SWL2-Internship/2026-1-swarm-vfi-model.git
    cd 2026-1-swarm-vfi-model
    ```

    Later, `git pull` brings in updates.

=== "Without git"

    ```bash
    wget https://github.com/DSO-ES-SWL2-Internship/2026-1-swarm-vfi-model/archive/refs/heads/master.zip
    unzip master.zip
    cd 2026-1-swarm-vfi-model-master
    ```

    To update later, download it again. (`sudo apt install unzip wget` if either is missing.)

The rest of this guide uses paths inside this folder, the *repository folder*.

### Python environment

```bash
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

## 2. The Raspberry Pi

The Pi only streams a video file, so it needs very little.

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

Put the sender and the video in `~/vfi` on the Pi, in this layout (the sender finds the video relative to its own
location):

```
vfi/streaming/scripts/sender.py
vfi/streaming/scripts/run_sender.sh
vfi/streaming/videos/sintel_trailer-480p.mp4
```

=== "With git"

    On the Pi:

    ```bash
    sudo apt install git          # if it is not installed yet
    git clone https://github.com/DSO-ES-SWL2-Internship/2026-1-swarm-vfi-model.git ~/vfi
    mkdir -p ~/vfi/streaming/videos
    ```

    The video is not in git, so copy it from the VM (or download it on the Pi with `wget`, from the link above):

    ```bash
    scp streaming/videos/sintel_trailer-480p.mp4 pi@172.20.10.2:vfi/streaming/videos/
    ```

    Later, `cd ~/vfi && git pull` brings in updates.

=== "Without git"

    From the VM, in the repository folder, copy every file with `scp`:

    ```bash
    ssh pi@172.20.10.2 mkdir -p vfi/streaming/scripts vfi/streaming/videos
    scp streaming/scripts/{sender.py,run_sender.sh} pi@172.20.10.2:vfi/streaming/scripts/
    scp streaming/videos/sintel_trailer-480p.mp4 pi@172.20.10.2:vfi/streaming/videos/
    ```

Get the Pi onto the same Wi-Fi as the board, then check from the VM: `ssh pi@172.20.10.2`.

## 3. The i.MX8 board

### 3a. The image

The board runs a custom Linux image built with Yocto. It already contains everything the relay needs: TensorFlow
Lite 2.19 with NXP's VX delegate (the NPU driver), GStreamer with the NXP plugins for the VPU and GPU, Python with
`numpy`, and Wi-Fi with auto-connect. You do not need to install any of it on the board.

- To **use** an existing SD card: set the boot switches to **0011** (SD card boot), insert the card, power on.
- To **build or change** the image: see [Yocto Setup](yocto.md).

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

Look for an `Inference (avg)` line and no errors. The **first** inference is much slower than the rest, because the NPU
compiles the model's graph. Later runs are fast.

### 3d. Copy the relay and the model

The board needs no video file: it receives the video from the Pi. It needs the relay scripts and the model.

=== "With git"

    The image includes git. The board needs internet access to reach GitHub. On the board:

    ```bash
    git clone https://github.com/DSO-ES-SWL2-Internship/2026-1-swarm-vfi-model.git ~/vfi
    ```

    That already includes the model (`artifacts/vfi_unet.tflite` is tracked in git, under 1 MB), so there is nothing else
    to copy. Later, `cd ~/vfi && git pull` brings in updates.

    If `~/vfi` already exists, because you copied files there before, rename it first (`mv ~/vfi ~/vfi-old`): `git clone`
    needs an empty or new folder.

=== "Without git"

    From the VM, in the repository folder, copy every file with `scp`:

    ```bash
    ssh root@172.20.10.10 mkdir -p vfi/streaming/scripts vfi/artifacts
    scp streaming/scripts/{relay.py,frame_dropper.py,hw_monitor.py,run_relay.sh,run_demo2.sh} \
        root@172.20.10.10:vfi/streaming/scripts/
    scp artifacts/vfi_unet.tflite root@172.20.10.10:vfi/artifacts/
    ```

Note: The relay finds the model at `../../artifacts/vfi_unet.tflite` relative to the script, which is why the folder layout
matters. It also keeps the compiled NPU graph in `artifacts/vx_cache/`, so the compile is paid only once.

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
| NPU runs the model | the `benchmark_model` command above | an `Inference (avg)` line, no errors |
| Relay works on its own | on the board: `cd ~/vfi/streaming/scripts && python3 relay.py toy_unet_tflite --input <trailer.mp4> --output fake` | the same number of frames in and out |

Then go to [Run the demos](run-the-demos.md).
