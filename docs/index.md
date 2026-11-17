# i.MX 8M Plus Hardware Utilisation using VFI

!!! question "Problem Statement"
    The i.MX 8M Plus has a CPU, a GPU, a video unit (VPU) and a neural network accelerator (NPU). 
    * How can the i.MX8M Plus's hardware resources be best used to improve payload availability?
    * How can software for the chosen application be developed?
    * How can the application be optimised for the hardware? 

 In order to demonstrate the utilisation of the hardware, the project builds a video streaming application (over multiple hops in a wireless network) utilising Video Frame Interpolation (VFI).

### The Setup
* Video Frame Interpolation (VFI) is a concept whereby additional intermediate frames are generated and inserted into the video playback.
* In the demo application, random frames are dropped, and VFI is used to insert replacements for these dropped frames.
* The VFI concept is realised using a machine learning application.
* This machine learning application is part of a bigger video streaming pipeline living on the i.MX8M Plus.
* The overall 3 device setup involves sending video data from the Raspberry Pi 4B to the i.MX8M Plus, where it is processed using VFI. Subsequently, the processed video stream will be sent to the VM where processed video and the video with dropped frames will be played side-by-side for the visual comparison.

### Project Focus
* The project focuses on the implementation of the application for utilisation on the hardware (as opposed to optimising the accuracy of VFI prediction).
* As such the documentation would largely focus on practical implementation details.

## The Hardware

The i.MX 8M Plus is a system-on-chip (SoC): several processing units on one chip, sharing the same memory through an
interconnect. This project uses four of them: the **CPU**, the **GPU**, the **VPU** and, as the focus, the **NPU**.

<div class="diagram" markdown="0">
--8<-- "assets/diagrams/soc.svg"
</div>

The Cortex-M7 and the HiFi 4 DSP are not used in this project.

### What each unit is for

| Unit | Designed for | Key facts | In this project |
| --- | --- | --- | --- |
| **NPU** | Neural network inference | 2.3 TOPS · fixed-point INT8 / INT16 only · faster with per-tensor quantisation · the first run compiles the network (can be cached) | Runs the VFI model (**the focus**) |
| **3D GPU** (GC7000UL) | 3D graphics; general compute | 2 shaders, 16 GFLOPS (FP32) · OpenGL ES 3.x, Vulkan, OpenCL 1.2 · runs neural networks in INT8 or FP16 | Scales the 480p video down to the model's input size |
| **VPU** | Video decoding and encoding | Decode 1080p60: H.265 (incl. 10-bit), H.264, VP9, VP8 (G1 + G2) · Encode 1080p60: H.264, H.265 (VC8000E) | Decodes the incoming H.264 video, and encodes the result |
| **CPU** (4× Cortex-A53) | General-purpose work; runs Linux | Up to 1.8 GHz · 32 KB + 32 KB L1 per core · 512 KB shared L2 · NEON SIMD | Everything that is not offloaded |
| **G2D** (2D GPU, GC520L) | 2D work: scaling, colour conversion, blending | A separate engine; shares a power domain and logic with the 3D GPU | Not used by the relay: the 3D GPU does the scaling |

Every unit shares one DRAM (32-bit LPDDR4-4000 or DDR4-3200, up to 8 GB, with inline ECC), which the NoC connects to all
of them. The specifications come from NXP's i.MX 8M Plus fact sheet, datasheet and Machine Learning User's Guide
(UG10166); see the
[credits](https://github.com/DSO-ES-SWL2-Internship/2026-1-swarm-vfi-model#credits-and-references).

### The NPU

The NPU is the part of the chip built for neural network inference, and it is the focus of this project. Three facts
about it shape everything else in these pages:

- **It works in fixed point only.** It computes with INT8 or INT16 numbers, not floating point. A model has to be
  quantised to run on it, and a float operation falls back to the CPU.
- **It is faster with per-tensor quantisation.** NXP recommends one scale per layer. See
  [the rules for the NPU](ml-model.md#export-it-for-the-npu).
- **The first run compiles the network.** The NPU compiles the model's graph the first time it runs, so the first
  inference is slow. The compiled graph can be cached.

To use the NPU, an application calls TensorFlow Lite and hands the work to the NPU through NXP's VX delegate. See
[Understanding the application](develop-on-board.md#understanding-the-application).

## The Demo

Three devices. 
* The Pi sends a video stream to the i.MX8.
* The i.MX8 board (the "relay") deliberately drops bursts of frames and recovers them using Video Frame Interpolation.
them with the model, and your computer shows the result next to the same video with the frames just held.

<div class="diagram" markdown="0">
--8<-- "assets/diagrams/demo-setup.svg"
</div>

Every step on the board is a hardware unit's job. The same relay can also run **entirely on the CPU**, to compare the two.

## Where to go next

### Demo

| Process | Helpful Link |
| --- | --- |
| Prepare the three devices for the demo | [Set up the three devices](setup.md) |
| See the demo working | [Run the demos](run-the-demos.md) |

### Development

Running a machine learning model on the i.MX8's NPU takes three ingredients:

1. **The model** you want to run.
2. **The environment** to run it in: the Linux image on the board, built with Yocto.
3. **The application** that calls the model and runs the inference on the NPU, in Python or C++.

With all three, you can run NPU inference on the i.MX8 for your own ML application. The GStreamer pipeline is one of the
main tools that *our* application uses to handle the video, but it is not the focus of the project, so it comes last.

| Process | Helpful Link |
| --- | --- |
| Build, retrain or export the model | [The model](ml-model.md) |
| Build or change the Linux image on the board | [The Yocto image](yocto.md) |
| Write and run my own Python or C++ on the board | [Developing on the board](develop-on-board.md) |
| Understand what the relay does with GStreamer | [The GStreamer pipeline](gstreamer-pipeline.md) |

### Tips

| Process | Helpful Link |
| --- | --- |
| Fix something that does not work | [Tips and troubleshooting](tips.md) |

## Credits

Training data: *Big Buck Bunny*, *Elephants Dream* and *Tears of Steel*, © Blender Foundation, Creative Commons
Attribution (Big Buck Bunny and Tears of Steel: CC BY 3.0); Vimeo-90K (Xue et al., IJCV 2019). The test and demo video
is the *Sintel* trailer, © Blender Foundation | durian.blender.org, CC BY 3.0. The films are not in this repository.
