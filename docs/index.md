# VFI on the i.MX 8M Plus

A video stream loses frames on the way. What if a small neural network could guess each missing frame from the frames
on either side of it? That is **video frame interpolation (VFI)**.

This project is not really about the network, though. It is a way to answer a different question:

!!! question "The question behind the project"
    The i.MX 8M Plus has a CPU, a GPU, a video unit (VPU) and a neural network accelerator (NPU). **Which one should do
    which job**, and how much does it matter? We picked a real task that uses all of them, and measured.

## The demo in one picture

Three devices. The Pi sends a video, the i.MX8 board (the "relay") deliberately drops bursts of frames and recovers
them with the model, and your computer shows the result next to the same video with the frames just held.

<div class="diagram" markdown="0">
--8<-- "assets/diagrams/demo-setup.svg"
</div>

Every step on the board is a hardware unit's job. The same relay can also run **entirely on the CPU**, which is how we
measured what the hardware buys us.

## What we found

All numbers were measured on the board on 16 November 2026, with the final model.

| Question | Answer |
| --- | --- |
| How long does the model take on the NPU? | **12.1 ms** per frame at 448×256. A 24 fps video leaves 41.7 ms per frame, so it fits with room to spare |
| ...on the CPU? | 490 ms on one core, 119 ms on all four. Too slow to keep up |
| ...on the GPU? | 9.5 seconds. About 20× *slower* than the CPU. The transposed convolution layers are the culprit |
| Does the way the model is quantised matter? | Yes: per-tensor INT8 is **1.9× faster** on the NPU than per-channel (12.1 vs 23.1 ms), for the same quality (within 0.02 dB) |
| What does the hardware buy for the whole relay? | On the 52-second trailer, flat out: **11 s** with the accelerators against **65 s** all on the CPU, using 1.2 cores instead of 3.8 |
| Does the model recover frames well? | On held-out Sintel clips: 36.3 dB PSNR against 35.6 dB for a plain blend of the two frames. On the perceptual score (LPIPS, lower is better) the plain blend is still slightly better, 0.063 against 0.072 |

!!! note "An honest reading of the last row"
    The model beats a simple blend on pixel accuracy but not on the perceptual score. It is a small model (70 thousand
    parameters) trained for 20 minutes on a CPU, built to fit the NPU, not to win benchmarks. Its job here is to be a
    realistic workload for the hardware.

## Where to go next

| I want to... | Read |
| --- | --- |
| Get the three devices ready | [Set up the three devices](setup.md) |
| See the demo working | [Run the demos](run-the-demos.md) |
| Understand, retrain or export the model | [The model](ml-model.md) |
| Understand what the relay does with GStreamer | [The GStreamer pipeline](gstreamer-pipeline.md) |
| Write and run my own Python or C++ on the board | [Developing on the board](develop-on-board.md) |
| Build or change the Linux image on the board | [The Yocto image](yocto.md) |
| Play with the ideas in the browser | [Visualisations](visualisations/index.md) |
| Fix something that does not work | [Tips and troubleshooting](tips.md) |

## What you need to know already

We assume you are comfortable with Linux, the shell, `ssh`/`scp` and git. Anything specific to this project
(Yocto, TensorFlow Lite, GStreamer, the NPU) is explained where it first appears.

## Credits

Training data: *Big Buck Bunny*, *Elephants Dream* and *Tears of Steel*, © Blender Foundation, Creative Commons
Attribution (Big Buck Bunny and Tears of Steel: CC BY 3.0); Vimeo-90K (Xue et al., IJCV 2019). The test and demo video
is the *Sintel* trailer, © Blender Foundation | durian.blender.org, CC BY 3.0. The films are not in this repository.
