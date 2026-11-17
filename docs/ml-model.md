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
add motion.

**Recipe.** Charbonnier loss (a smooth L1), Adam at 1e-3 with a cosine decay, 12 epochs of 250 steps of batch 8,
seed 0.

### Experiments

The choices behind the recipe (the output type, the data, the width, the loss, the seeds) were each tested one change at
a time. The runs are listed in `model/experiments/run_experiments.sh`, and their scores are written to
`artifacts/experiments/`.

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
    falls back to the CPU, so a model with float operations loses what the NPU offers.

    **2. Per-tensor weights, not per-channel.** TensorFlow quantises each output channel with its own scale by
    default. NXP says the NPU is optimised for one scale per layer (see its Machine Learning User's Guide, UG10166),
    so the export uses one scale per tensor.

    **3. Batch size fixed at 1.** A dynamic batch makes the transposed convolutions compute their output shape with a
    `STACK` operation. TIM-VX, the NPU's compiler, cannot compile it, and the delegate fails. You will see
    `Setup node STACK fail`.

    **4. The input scale must be exactly 1/255.** The export adds two sentinel pixels, 0 and 1, to the calibration
    data, so the input range is exactly 0–1. The relay can then feed raw 0–255 pixels with only a zero-point shift, with
    no float conversion. The relay checks this at start-up and refuses a model that breaks it.

The output scale is set by calibration and is not exactly 1/256, so the relay turns the model's INT8 output into pixels
with a 256-entry lookup table.

## Checking the speed

A 24 fps video leaves 41.7 ms per frame, so the model has to run in well under that to fit in a pipeline. Time it on the
board with `benchmark_model`: see [Developing on the board](develop-on-board.md#measuring-a-model-benchmark_model).
Re-time it after any change to the model or its export, and note the frame size, because the time depends on the
number of pixels. The first run after a change is much slower than the rest, because the NPU compiles the graph.

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
