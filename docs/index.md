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

| Process | Helpful Link |
| --- | --- |
| Get the three devices ready | [Set up the three devices](setup.md) |
| See the demo working | [Run the demos](run-the-demos.md) |
| Understand, retrain or export the model | [The model](ml-model.md) |
| Understand what the relay does with GStreamer | [The GStreamer pipeline](gstreamer-pipeline.md) |
| Write and run my own Python or C++ on the board | [Developing on the board](develop-on-board.md) |
| Build or change the Linux image on the board | [The Yocto image](yocto.md) |
| Fix something that does not work | [Tips and troubleshooting](tips.md) |

## Credits

Training data: *Big Buck Bunny*, *Elephants Dream* and *Tears of Steel*, © Blender Foundation, Creative Commons
Attribution (Big Buck Bunny and Tears of Steel: CC BY 3.0); Vimeo-90K (Xue et al., IJCV 2019). The test and demo video
is the *Sintel* trailer, © Blender Foundation | durian.blender.org, CC BY 3.0. The films are not in this repository.
