# The GStreamer pipeline

The relay (`streaming/scripts/relay.py`) is built with **GStreamer**, a library for building video processing
pipelines out of small pieces called *elements*. Each element does one job (decode, scale, encode) and passes
buffers (here, video frames) to the next. You describe the chain, and GStreamer moves the data.

You do not need to know GStreamer to run the demo. You do need it if you want to change what the relay does or
understand why it is shaped the way it is.

## Two pipelines, joined by Python

The relay is **two pipelines**, not one. Python code sits between them: that is where frames are dropped and the
model runs.

<div class="diagram" markdown="0">
--8<-- "assets/diagrams/relay-pipeline.svg"
</div>

**Why not one pipeline?** The model needs to look at *two* frames at once and sometimes produce *extra* frames. A
normal GStreamer chain passes one frame in, one frame out. With `appsink` (frames out of arm A, into Python) and
`appsrc` (frames from Python into arm B) we get to do anything we like in between.

## Arm A: receive and prepare

| Element | What it does | Why |
| --- | --- | --- |
| `tcpclientsrc` | Connects to the Pi and reads the byte stream | The Pi is the server. With `--input` the relay reads a local MP4 with `filesrc` instead |
| `tsdemux` | Splits the MPEG-TS container into streams | TS survives a plain TCP byte stream: it has its own sync bytes. (With a file: `qtdemux`) |
| `h264parse` | Finds the frame boundaries in the H.264 data | The decoder needs whole frames |
| decoder | Turns H.264 into raw frames. `vpudec` (the VPU, hardware) or `avdec_h264` (CPU) | The first big offload |
| `queue` | Buffers frames and gives everything after it its own thread | See [Threads and queues](#threads-and-queues) |
| scaler | Shrinks the frame to the model's 448×256 | `imxvideoconvert_ocl` on the GPU, or `videocrop ! videoscale` on the CPU |
| `videoconvert` | Takes the **luma (Y) plane** as a grey image (`GRAY8`) | The model works on grey. The decoder's YUV frames already contain it, so this is a copy |
| `appsink` | Hands each frame to Python | The way out of the pipeline |

### Why the scaler stays in YUV

The model takes **video-range grey** (16–235). That is exactly the Y plane of the decoder's YUV output. If the scaler
kept YUV, the converter after it only has to take the Y plane, with no colour maths and no range fix. That is what
`imxvideoconvert_ocl` does (NV12 in, NV12 out), at the lowest CPU cost of the options we measured.

### When is cropping needed?

The VPU decoder **pads its frames** to a multiple of 16 (for 480p, 854 → 864 columns) and marks the real picture with
crop information. Only the VPU decoder does this: `avdec_h264` outputs the picture at its true size.

- The **GPU scaler honours the crop information**, so nothing extra is needed.
- The **CPU scaler (`videoscale`) does not**, and would scale the padding into the picture. So for the CPU scaler the
  relay puts a `videocrop` in front, and sets it from the first caps: the decoder's frame size minus the stream's real
  picture size. With the CPU decoder the difference is zero, and the crop does nothing.

| Decoder | Scaler | Crop |
| --- | --- | --- |
| VPU | GPU (the default) | not needed, the GPU scaler handles it |
| VPU | CPU | a real crop, computed at start-up |
| CPU | CPU | present but set to 0 |

None of the three demos uses a real crop. It only matters for the "VPU decode, CPU scale" benchmark.

## The middle: Python

This is `RandomLossRecovery.on_new_sample` in `relay.py`. It runs once for every frame that arrives at the `appsink`.

1. **Maybe drop the frame.** `RandomFrameDropper` decides, to simulate a lossy network. A dropped frame is simply not
   passed on.
2. **Look for a gap.** Every frame has a timestamp. If the gap to the previous *kept* frame is more than one frame
   period (1/24 s), frames are missing. The gap tells us how many.
3. **Fill the gap** by repeated halving (see below): each missing frame is the model's guess at the midpoint of two
   existing frames, given its own timestamp.
4. **Push everything in order** into arm B's `appsrc`: first the recovered frames, then the kept one.

Between the two arms, the model sees uint8 pixels and turns them into INT8 inputs with a zero-point shift, stacks the
two frames as two channels, runs the NPU, and turns the INT8 output back into pixels through a 256-entry lookup table.
[The model](ml-model.md#export-it-for-the-npu) explains why that is exact.

!!! tip "Try it"
    The [frame loss visualisation](visualisations/frame-loss.md) lets you set the drop probability and the longest
    burst, and shows which frames are made from which.

## Arm B: encode and send

| Element | What it does | Why |
| --- | --- | --- |
| `appsrc` | Takes the frames Python pushes in | The way into the pipeline. `block=True`: if arm B falls behind, the model waits, instead of buffering without limit |
| `queue` | Buffering and a thread | |
| `videoconvert` → `I420` | Puts the grey frame into the luma plane, with neutral colour | Forced to `I420` so the encoder never converts it with its own range handling |
| encoder | `vpuenc_h264` (VPU), else `x264enc`, else `avenc_mpeg4`, all at about 2 Mbit/s | The second offload |
| `mpegtsmux` | Wraps the H.264 in MPEG-TS | Same container as the Pi's stream |
| `tcpserversink` | Waits for the VM to connect, then sends | `sync=True`: paced to the frames' timestamps, so the video plays in real time |

### Why does the output have its own frame rate?

Encoders need the real frame rate. Without one, `avenc_mpeg4` assumes 25 fps and drops frames that collide on that
grid. The relay reads the frame rate from the caps that reach the `appsink` (24/1) and gives it to `appsrc`.

## Threads and queues

A GStreamer pipeline runs in one thread unless something splits it. A `queue` does: everything after it runs in its
own thread. So with the queues, **decoding, scaling and the model run in parallel**, each working on a different
frame, instead of one after the other.

```text
without queues:   [decode][scale][model][encode]  [decode][scale][model][encode] ...   time per frame = SUM of the steps
with queues:      [decode]  [decode]  [decode]
                         [scale]  [scale]  [scale]                                     time per frame = the SLOWEST step
                                [model]  [model]
```

This is the idea behind the offloading numbers: a faster step helps only if it was the slowest one. See the
[pipeline budget visualisation](visualisations/pipeline-budget.md) to try it.

`appsrc` already hands frames to arm B on its own thread, so the model and the encoder run on different threads even
with `--no-queues`. The relay's stats line reports this as `shared_thread: false`.

## Options

```bash
python3 relay.py toy_unet_tflite --host <pi ip> --drop-probability 0.3 --max-drop-run 3
```

| Option | Values | What it picks |
| --- | --- | --- |
| first argument | `passthrough`, `frame_hold`, `linear_blend`, `toy_unet_tflite` | How lost frames are filled. The last one is the model |
| `--decoder` | `auto`, `cpu`, `vpu` | `auto` = VPU if present |
| `--scaler` | `auto`, `cpu`, `ocl` | `auto` = GPU (OpenCL) after the VPU decoder, else CPU |
| `--encoder` | `auto`, `vpu`, `x264`, `mpeg4` | `auto` = VPU, else x264, else MPEG-4 |
| `--inference` | `npu`, `cpu` | Where the model runs |
| `--input FILE` | an MP4 | Read a file instead of the Pi's stream |
| `--output` | `vm`, `fake`, `file` | `fake` discards the output and runs flat out. Use it to measure |
| `--no-queues` | | Leave out the queues, to see what they buy |
| `--drop-probability`, `--max-drop-run`, `--seed` | | The simulated frame loss |

When it finishes, the relay prints a `relay_stats` line with the wall time, the CPU time of the relay and of the whole
board, the model time per call, and the average time per frame in each step. `streaming/experiments/relay_bench.py`
runs the relay with one offload added at a time and tabulates those numbers.

## Gotchas we hit

- **The demuxer's output pad appears late.** `tsdemux` only creates its output once it has parsed the stream, so the
  link to `h264parse` is made in a `pad-added` callback.
- **`appsink` delivers the first frame twice** (once as a preroll sample, then as a normal one). Using both pushed
  frame 0 twice and broke the timestamps, so the preroll copy is discarded.
- **A missing timestamp is an error.** The gap detection needs a timestamp on every frame.
- **Do not trust a frame count on the VM side.** The viewer's live panel is not clocked, so a frame count says nothing
  about how smooth the video is. The viewer shows none for that reason.
