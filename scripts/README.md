# scripts/

Training, export and evaluation of the VFI U-Net, and the hardware measurements behind the project. The streaming
pipeline (Pi -> iMX8 -> VM) lives in `../streaming/`.

The scripts import each other (`model.py` holds the shared paths), so run them from this folder: `python3 <script>.py`.
The trained model is `../artifacts/vfi_unet.keras` and its INT8 export `../artifacts/vfi_unet.tflite`, which the
relay loads. Both are tracked in git (under 1 MB together).

## The model

The U-Net takes the frames before and after (2 channels, 448x256, grey, video range) and outputs the average of the two
plus a learned correction (`model.py`). Trained on frames of open Blender films (more motion) mixed with Vimeo-90K
triplets; scored on held-out Vimeo triplets and on clips of the Sintel trailer (the demo's own film, never trained on).

| Script | What it does |
| --- | --- |
| `model.py` | The architecture, the loss, and the paths of the trained model and its export |
| `data.py` | Vimeo-90K and Blender loaders, the augmentation, and the test sets |
| `blender_triplets.py` | Builds the Blender training windows from the films in `datasets/blender/` (needs ffmpeg; see below) |
| `train.py` | Trains the model; the defaults are the deployed recipe, the flags are the ablations (see its docstring) -> `artifacts/vfi_unet.keras` |
| `export_tflite.py` | Full-INT8 per-tensor export for the NPU, checked against float -> `artifacts/vfi_unet.tflite` |
| `evaluate.py` | PSNR, SSIM and LPIPS on the test sets, beside frame hold and linear blend |
| `experiments/run_experiments.sh` | The runs behind the choices (output type, data, seeds, width, loss), one change at a time |

Training data: download the three films to `datasets/blender/{bbb,ed,tos}/` (Big Buck Bunny 480p, Elephants Dream 480p,
Tears of Steel 720p, all from download.blender.org, CC BY, (c) Blender Foundation), then `python3 blender_triplets.py`
decodes them (about 90 s) into `datasets/blender/frames/` (5 GB, can be deleted: it is rebuilt from the films) and picks
the training windows. Vimeo-90K triplets go in `datasets/vimeo90k/`; the train/val/eval split lists are `../splits/`.
Evaluation needs `../streaming/videos/sintel_trailer-480p.mp4`.

## Experiments and hardware measurements

| Script | What it does |
| --- | --- |
| `experiments/compare_quantization.py` | fp32 vs INT8 per-channel vs INT8 per-tensor of the deployed model on the test sets |
| `experiments/gpu_layer_test.py` | Single-layer INT8 models at the U-Net's real shapes (and the whole model, per-channel and per-tensor), to find which layer makes the GPU slow -> `artifacts/gpu_layer_test/` |
| `experiments/export_float_variants.py` | FP32 and FP16 exports of the U-Net, its transposed convolution and MobileNet v1, to test whether the GPU runs float faster than INT8 -> `artifacts/float_variants/` |
| `experiments/benchmarks/benchmark_backends.sh [model.tflite ...]` | On the iMX8: CPU vs NPU vs GPU average inference time (default: the deployed model) -> `outputs/benchmarks/` |
| `experiments/benchmarks/plot_gpu_layers.py <results.txt> <out.png>` | Plots the single-layer times from `benchmark_backends.sh` |
| `cpp/` | C++ inference test (cross-compiled with the Yocto SDK): the same model through TFLite's C++ API |

Re-run `experiments/compare_quantization.py` (accuracy) and `experiments/benchmarks/benchmark_backends.sh` (speed) whenever the model or its
export settings change. Video codec and pipeline benchmarks are in `../streaming/experiments/`.
