# The model

The model is a small U-Net that looks at two video frames and predicts the one **between** them. This page explains
the idea, how it was trained, and, most importantly, the rules the model must follow to run on the NPU.

## The idea: a blend plus a correction

The simplest way to guess a missing frame is to average the frame before and the frame after. That is a *linear blend*.
It is sharp where nothing moves and ghosted where something does.

The model does not predict the frame from scratch. It starts from the blend and learns only the **difference**:

<div class="diagram" markdown="0">
--8<-- "assets/diagrams/blend-correction.svg"
</div>

That is `output = clip((A + B) / 2 + delta, 0, 1)`.

Two details matter:

- **`delta` starts at zero.** Before any training, the model is exactly a plain blend. It can only improve from there,
  and training only has to learn what the blend gets wrong.
- **The average is a 1×1 convolution, not a slice.** The two frames are two input channels. A 1×1 convolution with both
  weights fixed at 0.5 averages them. Because it is just another convolution, the whole graph stays on the NPU.

The model has about 70 thousand parameters. It works on **448×256 grey** frames in **video range** (16–235), which is
what the relay's decoder gives it, so no range conversion is needed.

## Train it

Everything is in the `model/` folder. Run the scripts from there.

```bash
cd model
python3 blender_triplets.py   # once: decodes the films and picks training windows (about 90 s)
python3 train.py              # -> ../artifacts/vfi_unet.keras (about 20 min on a CPU)
```

**Data.** 80% frames from open Blender films, which have more motion, and 20% Vimeo-90K triplets. The Blender films
are *Big Buck Bunny* and *Elephants Dream* (480p) and *Tears of Steel* (720p), from download.blender.org. Training
windows take frames `t, t+g, t+2g` for gaps `g = 1, 2`, with flips and time reversal. Why gaps of 2? Because the
demo fills a gap of four frames by halving, so the model needs to be good at midpoints two frames apart too.

**Why mix in Blender?** Vimeo-90K has very little motion, so a model trained only on it learns to blend. The films
add motion. See the table below for what it bought.

**Recipe.** Charbonnier loss (a smooth L1), Adam at 1e-3 with a cosine decay, 12 epochs of 250 steps of batch 8,
seed 0.

### What we tried

Each change was tested alone, and scored on 120 held-out clips from the *Sintel* trailer (frames 1 apart). The
earlier model and the linear blend are there for reference.

| Run | What it is | PSNR | LPIPS |
| --- | --- | --- | --- |
| Earlier model | Plain output, trained on Vimeo only, different recipe | 30.88 dB | 0.149 |
| a0 | Plain sigmoid output, Vimeo only (the control) | 33.24 dB | 0.100 |
| a1 | a0 + Blender films | 33.57 dB | 0.076 |
| a2 | Blend + correction, Vimeo only | 36.62 dB | 0.081 |
| **a3** | **Blend + correction + Blender films (deployed)** | 36.31 dB | **0.072** |
| a4 | a3 with 1.5× the channels | 36.37 dB | 0.074 |
| a5 | a3 + a sharpness (gradient) loss | 36.62 dB | 0.078 |
| a6 | a3 with 2× the channels | collapsed to the plain blend | |
| (baseline) | Linear blend of the two frames | 35.59 dB | 0.063 |

What the table says:

- **The blend-plus-correction output is the big win**, about +3 dB (a0 → a2, a1 → a3).
- **The Blender films mainly help the perceptual score**: LPIPS drops from 0.081 to 0.072 (a2 → a3), for 0.3 dB of
  PSNR. That is why a3 is deployed and not a2.
- **A wider network and a sharpness loss did not help.** The sharpness loss raises PSNR back to 36.62 dB but makes
  LPIPS worse.
- **a6 is a failed run, not evidence.** It collapsed to the plain blend (its scores are identical to it), so the
  width conclusion rests on a4.
- **Check the noise before believing a gain.** Rerunning with another seed moved the plain model by 0.8 dB, and a3 by
  only 0.05 dB. A change smaller than that is not a result.
- **No version of the model beats the plain blend on LPIPS.** The model wins on PSNR (0.7 dB) and loses on LPIPS.

The runs are in `model/experiments/run_experiments.sh`, and the scores in `artifacts/experiments/`.

## Evaluate it

```bash
python3 evaluate.py           # PSNR, SSIM and LPIPS against frame hold and linear blend
```

| Metric | What it tells you |
| --- | --- |
| **PSNR** (dB, higher is better) | Are the pixel values close? |
| **SSIM** (0–1, higher is better) | Does the picture look structurally the same? |
| **LPIPS** (lower is better) | Do the two images look alike to a neural network trained on human judgements? |

It scores on 40 held-out Vimeo triplets and on 120 clips each of frames 1 and 2 apart from the *Sintel* trailer, the
demo's own film, which is **never trained on**.

## Export it for the NPU

```bash
python3 export_tflite.py      # -> ../artifacts/vfi_unet.tflite (about 80 KB)
```

The NPU is picky. The export script enforces each of these rules, and each one has a reason:

!!! abstract "The four rules"
    **1. Fully INT8, input and output too.** The NPU only does fixed-point arithmetic. Any float operation silently
    falls back to the CPU. We measured the float model on the "NPU": **5.6 seconds** per frame.

    **2. Per-tensor weights, not per-channel.** TensorFlow quantises each output channel with its own scale by
    default. NXP says the NPU is optimised for one scale per layer. Measured: **12.1 ms against 23.1 ms**, 1.9×
    faster, for about 0.02 dB difference in quality.

    **3. Batch size fixed at 1.** A dynamic batch makes the transposed convolutions compute their output shape with a
    `STACK` operation. TIM-VX, the NPU's compiler, cannot compile it, and the delegate fails. You will see
    `Setup node STACK fail`.

    **4. The input scale must be exactly 1/255.** The export adds two sentinel pixels, 0 and 1, to the calibration
    data, so the input range is exactly 0–1. The relay can then feed raw 0–255 pixels with only a zero-point shift, with
    no float conversion. The relay checks this at start-up and refuses a model that breaks it.

The output scale is set by calibration and is not exactly 1/256, so the relay turns the model's INT8 output into pixels
with a 256-entry lookup table. That is cheaper than a float rescale on the board's A53 cores.

### How much does INT8 cost in quality?

Measured on the Sintel clips (frames 1 apart), PSNR / SSIM:

| Model | PSNR | SSIM | NPU time |
| --- | --- | --- | --- |
| 32-bit float | 36.31 dB | 0.9334 | not meaningful (5.6 s) |
| INT8 per-channel | 36.03 dB | 0.9306 | 23.1 ms |
| INT8 per-tensor (deployed) | 36.05 dB | 0.9307 | **12.1 ms** |

Going to INT8 costs about 0.25 dB. Per-tensor against per-channel makes no real difference.

## Speed on the board

| Frame size | NPU time | First run (graph compile) |
| --- | --- | --- |
| 448×256 | 12.1 ms | 28 s |
| 640×360 | 25.0 ms | 52 s |
| 848×480 | 43.6 ms | 77 s |
| 1280×720 | 101.3 ms | 140 s |

The time grows with the number of pixels, at about 110 ms per megapixel. The budget for 24 fps is **41.7 ms**, so 480p
is just over. See how this plays out in the [pipeline budget visualisation](visualisations/pipeline-budget.md).

## Where things are

| File | What it does |
| --- | --- |
| `model/model.py` | The architecture, the loss, and the paths of the trained model and its export |
| `model/data.py` | Vimeo-90K and Blender loaders, the augmentation, and the test sets |
| `model/train.py` | Training. The defaults are the deployed recipe, the flags are the experiments |
| `model/export_tflite.py` | The INT8 export, with the NPU rules above |
| `model/evaluate.py` | PSNR, SSIM and LPIPS |
| `model/experiments/` | The runs and hardware measurements behind the choices. See its README |
| `artifacts/vfi_unet.keras`, `vfi_unet.tflite` | The deployed model (under 1 MB together, tracked in git) |
