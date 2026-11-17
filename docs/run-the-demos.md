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
    compiled. The first run after a model change is slow, because the NPU compiles the model.

## Demo 1: the full pipeline

Start the three parts **in this order**. The relay waits for the viewer before it connects to the Pi.

=== "1. Pi"

    ```bash
    cd ~/vfi/streaming/scripts
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

To run it again, restart the sender and the relay (they exit when the video ends), then press **Connect**.

!!! tip "The processed video does not look good."
    This is expected as the focus of the demonstration is show that the application functions, and in the later demos, to show hardware utilisation and its effects. This is as opposed to optimising the model accuracy to produce a sharp and accurate reconstruction of dropped frames. Additionally, the model trained and deployed is a lightweight (i.e. small) model that is trained on a limited dataset.


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
- It takes longer than the 52 s the trailer lasts.

Everything runs on the CPU in this mode: software decode, software scale, the model on four threads, and a software
encoder (MPEG-4).

## Demo 3: with the hardware units

Same as Demo 2, but start the relay with:

```bash
./run_demo2.sh hw
```

**What you should see**

- The video stays smooth and keeps up with real time.
- Fewer CPU cores are busy than in Demo 2.
- The monitor shows the VPU (decode and encode) and the NPU working.

## Reading the load monitor

`hw_monitor.py` refreshes four times a second. Each row is a one-second average.

| Row | Meaning |
| --- | --- |
| CPU 0–3 | Load of each of the four A53 cores |
| GPU | GPU load, from the kernel's debug filesystem |
| NPU | NPU load |
| VPU decode / encode | Frames per second through each video unit |

!!! note "The GPU and NPU rows are not accurate utilisation percentages"
    Based on preliminary investigation, they count idle cycles reported by the driver, rather than the utilisation of the compute resources in these hardware elements. Use them to see *whether* a unit is working, not to say it is
    "60% used".

If a row stays at zero when it should not, check that debugfs is mounted (`mount -t debugfs none /sys/kernel/debug`).

## If something goes wrong

See [Tips](tips.md).
