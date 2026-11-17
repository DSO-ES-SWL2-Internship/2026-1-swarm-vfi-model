# Tips and troubleshooting

Most problems on this project fall into a handful of groups. Find your symptom first.

## The board does not answer SSH

The Wi-Fi did not connect by itself on boot. Use the serial console:

1. Close any other program using the port (PuTTY, for one). Two readers garble each other.
2. Open the board's debug USB port: `/dev/ttyUSB2`, 115200 baud, 8N1. Log in as `root` (no password).
3. Run `wpa_cli -i mlan0 scan`, then `udhcpc`. SSH should answer after that.
4. If `mlan0` does not exist at all, the driver did not load. See the Wi-Fi section of [The Yocto image](yocto.md).

## The demo does not start

| Symptom | Cause and fix |
| --- | --- |
| The relay prints nothing after "Listening" | It waits for the viewer. Press **Connect** (or use `--connect`) |
| The viewer connects but no video appears | The sender must be running first. Start in this order: Pi, board, VM |
| "Address already in use" on port 5001 | A previous relay is still running. `pkill -f relay.py` |
| "Missing local video(s)" in the viewer | `streaming/videos/stuttering_burst.mp4` is missing. Make it with `make_burst_demo.sh -d` |
| "Unsupported timestamp gap" | The relay expects 24 fps input, with gaps of at most `max_drop_run + 1` frames |
| The stutter on the left does not match the right | The baseline was made with other drop settings. Make it again with the relay's values (0.3, 3, seed 42) |

!!! warning "Is it the right copy of the code?"
    If the board has an older clone of the repository, running from there means running an old relay with an old
    model. Always start from `~/vfi/streaming/scripts`. The relay prints which model it loads: look for
    `vfi_unet.tflite`.

## The model is slow, or the NPU is not used

| Symptom | Cause and fix |
| --- | --- |
| The first frame takes a long time | The NPU is compiling the graph. Normal. It is cached afterwards. **Run it once before you present** |
| The model is far slower than expected | A float operation made it fall back to the CPU. Check the export: fully INT8, input and output too |
| `Setup node STACK fail` | The model has a dynamic batch size. Re-export with the batch fixed at 1 |
| "input scale is not 1/255; re-export" | The relay refuses a model whose input range is not exactly 0–1. Export with `export_tflite.py`, which adds the sentinel pixels |
| It runs, but at CPU speed | The delegate was not loaded. Is `/usr/lib/libvx_delegate.so` there? The relay says "running on the NPU" or "running on CPU" when it starts |

## The load monitor shows zeros

Mount debugfs (once per boot): `mount -t debugfs none /sys/kernel/debug`. The GPU and NPU rows count idle cycles, so
read them as "is it working", not as a percentage.

## Python and TensorFlow

- **Import `torch` before `tensorflow`.** The other way round crashes the process. `model/evaluate.py` does this for you.
- The code targets **Python 3.10**. In an f-string, you cannot use a backslash inside the braces.
- Training takes about 20 minutes on a CPU. Run long jobs with `setsid nohup` so they survive a closed terminal.

## Habits that saved time

- **Check the board with `benchmark_model` first** after any change to the model or the image. It takes a minute and
  tells you whether the NPU works.
- **Change one thing at a time**, both when training and when measuring.
- **Measure before you believe.** Time a change before you assume it helps.
- **Write down the settings with every number.** Frames per second means little without the resolution, the delegate
  and the number of threads.
