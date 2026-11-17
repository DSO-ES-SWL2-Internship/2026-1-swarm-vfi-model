# experiments/ (model)

The runs behind the model choices and the hardware measurements. They are kept so a number can be re-run; the demo does
not use them. Run from this folder's parent (`model/`) or from here: each script finds the modules in `model/`.

## Model quality (on the VM)

| Script | What it answered |
| --- | --- |
| `run_experiments.sh` | Which changes improve the model, one at a time: output type (plain / residual) x data (Vimeo / Blender + Vimeo), then seeds, width and loss. The deployed model is its run `a3`: `train.py`'s defaults. Scores in `artifacts/experiments/results.txt` |
| `compare_quantization.py` | What per-tensor INT8 (1.9x faster on the NPU) costs in quality against per-channel and fp32 |

## Hardware (export on the VM, time on the iMX8)

| Script | What it answered |
| --- | --- |
| `gpu_layer_test.py` | Which layer makes the GPU slow: single-layer INT8 models at the U-Net's real shapes (and the whole model), per-channel and per-tensor -> `artifacts/gpu_layer_test/` |
| `export_float_variants.py` | Does the GPU run float faster than INT8? FP32 and FP16 exports of the U-Net, a transposed convolution and MobileNet v1 -> `artifacts/float_variants/` |
| `benchmarks/benchmark_backends.sh [model.tflite ...]` | CPU vs NPU vs GPU average inference time with `benchmark_model` (default: the deployed model) -> `outputs/benchmarks/` |
| `benchmarks/plot_gpu_layers.py` | Plots the single-layer times from `benchmark_backends.sh` |

Measured on the deployed model (16 Nov 2026): on the NPU, per-channel 23.07 ms and per-tensor 12.12 ms (1.9x); the
whole U-Net on the GPU 9.5 s per-channel and 10.4 s per-tensor, 20x slower than the CPU (490 ms on one thread), with the
transposed convolution (6.0 s) the slowest layer. Quality: per-tensor and per-channel are within 0.02 dB PSNR on the
Sintel clips, and INT8 costs about 0.25 dB against fp32. Re-run these scripts after any change to the model or its export.
