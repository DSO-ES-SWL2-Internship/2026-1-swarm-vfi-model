# scripts/

Model training, export, evaluation and hardware benchmarks for the toy VFI U-Net.
The streaming pipeline (Pi -> iMX8 -> VM) lives in `../streaming/`.

The Python scripts import each other (`training.py` holds the shared constants and paths),
so run them from this folder: `python3 <script>.py`.

## Model (run on the VM)

| Script | What it does | Output |
| --- | --- | --- |
| `vimeo_dataset.py` | Loads Vimeo-90K triplets and the train/val/eval split manifests; imported by the others | – |
| `training.py` | Trains the toy U-Net (`--data vimeo` or synthetic circles) | `artifacts/toy_unet.keras` |
| `export_tflite.py` | Full-INT8 TFLite export for the NPU (`PER_TENSOR = True`: 1.9x faster on the NPU, -0.02 dB) | `artifacts/toy_unet_int8.tflite` |
| `export_onnx.py` | INT8 ONNX export (ONNX/VSINPU route turned out to be a dead end on the board; kept for reference) | `artifacts/toy_unet_int8.onnx` |

## Evaluation (run on the VM)

| Script | What it does | Output |
| --- | --- | --- |
| `interpolate.py` | Runs every technique (frame hold, linear blend, U-Net fp32 and INT8) on Vimeo / synthetic examples | `outputs/<source>/<id>/` |
| `eval.py` | PSNR / SSIM / LPIPS over `interpolate.py`'s outputs, plus box plots | `artifacts/metrics_boxplot.png` |
| `visual_grid.py` | Side-by-side picture of each technique on one example | `artifacts/visual_grid_<source>.png` |
| `compare_quantization.py` | fp32 vs INT8 per-channel vs INT8 per-tensor accuracy on the same eval triplets | printed table |

Re-run `compare_quantization.py` (accuracy) and `benchmarks/benchmark_backends.sh` (speed) whenever the
model or its export settings change.

## Hardware benchmarks

| Script | Runs on | What it does |
| --- | --- | --- |
| `gpu_layer_test.py` | VM | Builds single-layer INT8 models (per-channel and per-tensor) to find which layer makes the GPU slow -> `artifacts/gpu_layer_test/` |
| `benchmarks/benchmark_backends.sh [model.tflite ...]` | iMX8 | CPU vs NPU vs GPU average inference time for any TFLite models (default: the deployed U-Net) -> `outputs/benchmarks/` |

Video codec benchmarks (VPU, G2D, CPU codecs, utilisation) are in `../streaming/scripts/test/codec_suite.py`.
