# i.MX 8M Plus Hardware Utilisation with VFI

### Overview
* This project explores the utilisation of the hardware resources (i.e. NPU, GPU and VPU) on the i.MX8M Plus.
* In order to demonstrate the utilisation of the hardware, the project builds a video streaming application (over multiple hops in a wireless network) utilising Video Frame Interpolation (VFI).

### The Setup
* Video Frame Interpolation (VFI) is a concept whereby additional intermediate frames are generated and inserted into the video playback.
* In the demo application, random frames are dropped, and VFI is used to insert replacements for these dropped frames.
* The VFI concept is realised using a machine learning application.
* This machine learning application is part of a bigger video streaming pipeline living on the i.MX8M Plus.
* The overall 3 device setup involves sending video data from the Raspberry Pi 4B to the i.MX8M Plus, where it is processed using VFI. Subsequently, the processed video stream will be sent to the VM where processed video and the video with dropped frames will be played side-by-side for the visual comparison.

### Project Focus
* The project focuses on the implementation of the application for utilisation on the hardware (as opposed to optimising the accuracy of VFI prediction).
* As such the documentation would largely focus on practical implementation details.

## Documentation

The full guide (setting up the Pi, the board and the VM, running the demos, the model, the GStreamer pipeline, the
and the Yocto image) is at
**https://dso-es-swl2-internship.github.io/2026-1-swarm-vfi-model/**.

Its source is in `docs/`; preview it on your
machine with `pip install -r requirements-docs.txt && mkdocs serve`.

## Directory

| Folder | What is in it |
| --- | --- |
| `model/` | The model: architecture, training, INT8 export for the NPU, evaluation; `model/experiments/` the runs behind the choices; `model/cpp/` a C++ inference test. See `model/README.md` |
| `streaming/` | The three-device demo: sender, relay, viewer, and the frame-loss simulation; `streaming/experiments/` the pipeline benchmarks. See `streaming/README.md` |
| `artifacts/` | The deployed model: `vfi_unet.keras` (trained) and `vfi_unet.tflite` (INT8, what the relay loads). Everything else in it is generated |
| `splits/` | The Vimeo-90K train / val / eval triplet lists |

## Credits and references

The numbers in square brackets are the same as on the *Sources* slides of the talk.

### Data and video

Only the Sintel trailer is used as the test and demo video. None of the films or datasets is included in this
repository: you download them yourself (see `model/README.md`).

| | Work | Licence | Used for |
| --- | --- | --- | --- |
| [8] | T. Xue, B. Chen, J. Wu, D. Wei, W. T. Freeman, "Video Enhancement with Task-Oriented Flow", *International Journal of Computer Vision* 127, 2019 ([Vimeo-90K](http://toflow.csail.mit.edu/)) | See the dataset's terms | 20% of the training data, and the 40 held-out test triplets |
| [16] | *Big Buck Bunny* (Peach Open Movie Project, 2008), (c) Blender Foundation, [peach.blender.org](https://peach.blender.org/) | CC BY 3.0 | Training data |
| [17] | *Elephants Dream* (Orange Open Movie Project, 2006), (c) Blender Foundation, [orange.blender.org](https://orange.blender.org/) | CC BY 2.5 | Training data |
| [18] | *Tears of Steel* (Mango Open Movie Project, 2012), (c) Blender Foundation, [mango.blender.org](https://mango.blender.org/) | CC BY 3.0 | Training data |
| [9] | *Sintel* trailer (Durian Open Movie Project, 2010), (c) Blender Foundation, [durian.blender.org](https://durian.blender.org/), [download](https://download.blender.org/durian/trailer/) | CC BY 3.0 | The demo video and the held-out test clips (never trained on) |

### Papers and methods

| | Reference | What it is used for |
| --- | --- | --- |
| [6] | S. Liu, "ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors", arXiv:2603.26835, 2026. [arxiv.org/abs/2603.26835](https://arxiv.org/abs/2603.26835) | Inspiration: frame interpolation on mobile accelerators, with an INT8 U-Net on the NPU |
| [7] | O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015. [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597) | The U-Net architecture |
| [10] | A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", 2017. [arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861) | MobileNet v1, the reference model in the GPU comparison |
| [19] | R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018. [arxiv.org/abs/1801.03924](https://arxiv.org/abs/1801.03924) (code: [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), the `lpips` package) | LPIPS, the perceptual score |
| [20] | Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity", *IEEE Transactions on Image Processing* 13(4), 2004 | SSIM, the structural score (PSNR is the usual peak signal-to-noise ratio) |
| [21] | P. Charbonnier, L. Blanc-Feraud, G. Aubert, M. Barlaud, "Two Deterministic Half-Quadratic Regularization Algorithms for Computed Imaging", ICIP 1994 | The Charbonnier loss used in training |
| [22] | D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015. [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980) | The optimiser |
| [23] | B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018. [arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877) | Background for the INT8 quantisation that TensorFlow Lite applies |

### NXP hardware documentation and software

| | Reference | Used for |
| --- | --- | --- |
| [1] | NXP, *i.MX 8M Plus Applications Processor Family* fact sheet (IMX8MPLUSFS). [PDF](https://www.nxp.com/docs/en/fact-sheet/IMX8MPLUSFS.pdf) | Block diagram, GPU APIs |
| [2] | NXP, *i.MX 8M Plus Applications Processor Datasheet for Consumer Products* (IMX8MPCEC, Rev. 0, 03/2021) | The feature table: caches, GPU, VPU codecs, NPU |
| [3] | NXP, *i.MX Machine Learning User's Guide* (UG10166), section 8.1. [PDF](https://www.nxp.com/docs/en/user-guide/UG10166.pdf) | NPU fixed-point support, warm-up and graph caching, per-tensor against per-channel quantisation |
| [4] | Freescale, libimxvpuapi README. [GitHub](https://github.com/Freescale/libimxvpuapi/blob/master/README.md) | The VPU decoders and encoder |
| [5] | NXP, *i.MX DSP User's Guide* (UG10167). [PDF](https://www.nxp.com/docs/en/user-guide/UG10167.pdf) | The DSP (background) |
| [11] | NXP, *i.MX 8M Plus Applications Processor Reference Manual* (IMX8MPRM, Rev. 3, 08/2024). [PDF](https://www.nxp.com/docs/en/reference-manual/IMX8MPRM.pdf) | Interconnect, bus masters, DDR controller |
| [13] | NXP, [tflite-vx-delegate-imx](https://github.com/nxp-imx/tflite-vx-delegate-imx) | Runs the TensorFlow Lite model on the NPU, or on the GPU with `USE_GPU_INFERENCE=1` |
| [24] | NXP, i.MX Linux BSP release LF6.18.20_2.0.0 ([imx-manifest](https://github.com/nxp-imx/imx-manifest), [meta-imx](https://github.com/nxp-imx/meta-imx)), and the *i.MX Yocto Project User's Guide* (UG10164) | The base of the board's Linux image |
| [25] | NXP, [imx-docker](https://github.com/nxp-imx/imx-docker) | The Docker build environment our Yocto build is based on |

### Software

| | Software | Used for |
| --- | --- | --- |
| [12] | [TensorFlow](https://www.tensorflow.org/), Keras and TensorFlow Lite (2.19 on the board) | Training the model; the INT8 export; running and timing it |
| [14] | [GStreamer](https://gstreamer.freedesktop.org/) and PyGObject, with NXP's `imx-gst1.0-plugin` for the VPU, G2D and GPU | The video pipelines |
| [15] | [Yocto Project](https://www.yoctoproject.org/) and OpenEmbedded | Building the board's Linux image |
| | [PyTorch](https://pytorch.org/) and [`lpips`](https://github.com/richzhang/PerceptualSimilarity) | Computing LPIPS |
| | [FFmpeg](https://ffmpeg.org/) | Decoding video for training and evaluation |
| | [NumPy](https://numpy.org/), [Pillow](https://python-pillow.org/), [Matplotlib](https://matplotlib.org/) | Data handling, image loading, plots |
| | [MkDocs](https://www.mkdocs.org/) with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) | The documentation site |

The exact versions used for training are in `requirements.txt`.

### Our own work

The model, the training and export scripts, the relay, the frame-loss simulation, the viewer, the benchmarks and the
documentation in this repository are original to this project. Measurements are our own, taken on an i.MX 8M Plus EVK
(Raspberry Pi 4B as the sender).
