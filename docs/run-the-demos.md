# Run the demos

There are three demos. They all show the same thing, the Pi's video with random bursts of frames lost and recovered
by the model, but on different paths through the board:

| Demo | What it shows | Relay command |
| --- | --- | --- |
| **1. Full pipeline** | The whole setup works: lost frames are recovered | `./run_relay.sh` |
| **2. All CPU** | What happens *without* the hardware units | `./run_demo2.sh cpu` |
| **3. With hardware** | The same, using the VPU, GPU and NPU | `./run_demo2.sh hw` |

Demos 1 and 3 run the same path. Demo 1 is the plain version, and Demo 3 adds the load monitor so you can watch the
units work.

!!! info "Before you start"
    Everything in [Set up the three devices](setup.md) is done, and you have run the relay once so the NPU graph is
    compiled. The first run after a model change is slow (about 30 s).

## Demo 1: the full pipeline

Start the three parts **in this order**. The relay waits for the viewer before it connects to the Pi.

=== "1. Pi"

    ```bash
    cd <repo>/streaming/scripts
    ./run_sender.sh
    ```

    It sends the 24 fps Sintel trailer and waits for the relay to connect.

=== "2. i.MX8"

    ```bash
    cd ~/vfi/streaming/scripts
    ./run_relay.sh
    ```

    Wait for `Listening on 0.0.0.0:5001 for the VM to connect...`.

=== "3. VM"

    ```bash
    cd streaming/scripts
    python3 demo_viewer.py --connect
    ```

    A window opens with two panels. Without `--connect`, press **Connect** yourself.

**What you should see**

- **Left panel, "Without VFI":** the trailer with frames dropped and the previous frame held. It stutters.
- **Right panel, "With VFI":** the live stream from the board, with the same frames dropped and then recovered by the
  model. It is smoother.
- The trailer lasts about 52 seconds. Then the sender and the relay exit and the relay prints its summary:

```text
Dropped frames: 465 in 239 bursts      (37% of 1251 frames)
Burst lengths: 1x82, 2x88, 3x69
Recovered frames: 465
Mean inference latency (465 calls): about 13 ms/frame
```

To run it again, restart the sender and the relay (they exit when the video ends), then press **Connect**.

!!! tip "The two panels only roughly line up"
    The left video starts when the first recovered frame arrives on the right. They are close, but not frame-exact.
    That is fine for judging smoothness. It is not a measurement.

??? question "Why does the stutter match exactly?"
    The relay and the baseline video use the same drop settings: probability 0.3, bursts of up to 3 frames, seed 42.
    Same seed, same frames dropped. The baseline file `streaming/videos/stuttering_burst.mp4` was made with
    `make_burst_demo.sh -d` using those values. If you change the relay's settings, make a new baseline with the same
    ones. See the [frame loss visualisation](visualisations/frame-loss.md) to explore what these settings do.

## Demo 2: all on the CPU

Use **two windows on the board**: one for the relay, one for the load monitor.

=== "i.MX8, window 1"

    ```bash
    cd ~/vfi/streaming/scripts
    ./run_demo2.sh cpu
    ```

=== "i.MX8, window 2"

    ```bash
    cd ~/vfi/streaming/scripts
    python3 hw_monitor.py
    ```

Then start the sender on the Pi and press **Connect** in the viewer, as in Demo 1.

**What you should see**

- The video **lags**. The relay cannot keep up, so recovery falls behind.
- All four CPU cores sit near 100%. The NPU, GPU and VPU rows stay near 0.
- It takes about **65 s** for the 52 s trailer. The final summary shows about **133 ms** per model call.

Everything runs on the CPU in this mode: software decode, software scale, the model on four threads, and a software
encoder (MPEG-4).

## Demo 3: with the hardware units

Same as Demo 2, but start the relay with:

```bash
./run_demo2.sh hw
```

**What you should see**

- The video stays smooth and keeps up with real time (about 52 s).
- About **1.2 cores** busy, not 3.8.
- The monitor shows the VPU (decode and encode) and the NPU working.
- About **13 ms** per model call in the summary.

## Reading the load monitor

`hw_monitor.py` refreshes four times a second. Each row is a one-second average.

| Row | Meaning |
| --- | --- |
| CPU 0–3 | Load of each of the four A53 cores |
| GPU | GPU load, from the kernel's debug filesystem |
| NPU | NPU load |
| VPU decode / encode | Frames per second through each video unit |

!!! note "The GPU and NPU rows are not a utilisation percentage"
    They count idle cycles reported by the driver. Use them to see *whether* a unit is working, not to say it is
    "60% used".

If a row stays at zero when it should not, check that debugfs is mounted (`mount -t debugfs none /sys/kernel/debug`).

## If something goes wrong

See [Tips and troubleshooting](tips.md). The most common problems are the board's Wi-Fi not connecting, starting the
parts in the wrong order, and running the relay from an old copy of the code.
