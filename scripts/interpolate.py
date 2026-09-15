"""
Runs every interpolation technique (frame_hold, linear_blend, toy_unet) against a
data source and writes a uniform outputs/<source>/<id>/ folder per example, for
eval.py and visual_grid.py to consume. 
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow import keras

import training
import vimeo_dataset

OUTPUT_ROOT = training.PROJECT_ROOT / "outputs"

SYNTHETIC_COUNT = 5
SYNTHETIC_DISPLACEMENT_RANGE = (150, 300)  # wider than training's (10, 150): motion should be obvious to the eye

# Runs frame hold classical method
def frame_hold(frame0_img):
    out = frame0_img.copy()
    draw = ImageDraw.Draw(out)
    draw.text((10, 10), "frame hold", fill="white", stroke_width=2, stroke_fill="black")
    return out

# Runs linear blend classical method
def linear_blend(frame0_img, frame2_img):
    blended = Image.blend(frame0_img, frame2_img, alpha=0.5)
    draw = ImageDraw.Draw(blended)
    draw.text((10, 10), "linear blend", fill="white", stroke_width=2, stroke_fill="black")
    return blended


def load_model():
    return keras.models.load_model(
        training.MODEL_PATH,
        custom_objects={"weighted_mae": training.weighted_mae, "charbonnier_loss": training.charbonnier_loss},
    )

# Runs VFI using toy U-net model
def run_toy_unet(model, frame0_img, frame2_img):
    f0 = np.asarray(frame0_img, dtype=np.float32)[..., None] / 255.0
    f2 = np.asarray(frame2_img, dtype=np.float32)[..., None] / 255.0
    x = np.concatenate([f0, f2], axis=-1)[None, ...]
    pred = model.predict(x, verbose=0)
    pred_arr = (pred[0, ..., 0] * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pred_arr, mode="L")


def load_tflite_interpreter():
    interpreter = tf.lite.Interpreter(model_path=str(training.TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


# Runs VFI using the INT8-quantized TFLite export of the toy U-net model
def run_toy_unet_int8(interpreter, frame0_img, frame2_img):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    f0 = np.asarray(frame0_img, dtype=np.float32)[..., None] / 255.0
    f2 = np.asarray(frame2_img, dtype=np.float32)[..., None] / 255.0
    x = np.concatenate([f0, f2], axis=-1)[None, ...]

    in_scale, in_zero_point = input_details["quantization"]
    x_q = np.round(x / in_scale + in_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details["index"], x_q)
    interpreter.invoke()
    out_q = interpreter.get_tensor(output_details["index"])

    out_scale, out_zero_point = output_details["quantization"]
    pred = (out_q.astype(np.float32) - out_zero_point) * out_scale
    pred_arr = (pred[0, ..., 0] * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pred_arr, mode="L")


def array_to_gray_image(arr):
    """arr: float32 HxWx1 in [0, 1] (as produced by training.render_circle/make_triplet)."""
    img_arr = (arr.squeeze(-1) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_arr, mode="L")

# Save source and output images in their respective directories
def save_example(example_dir, frame0, ground_truth, frame2, model, interpreter):
    example_dir.mkdir(parents=True, exist_ok=True)
    frame0.save(example_dir / "frame0.png")
    ground_truth.save(example_dir / "ground_truth.png")
    frame2.save(example_dir / "frame2.png")
    frame_hold(frame0).save(example_dir / "frame_hold.png")
    linear_blend(frame0, frame2).save(example_dir / "linear_blend.png")
    run_toy_unet(model, frame0, frame2).save(example_dir / "toy_unet.png")
    run_toy_unet_int8(interpreter, frame0, frame2).save(example_dir / "toy_unet_int8.png")


def run_vimeo(model, interpreter):
    triplets = vimeo_dataset.load_manifest("eval.txt")
    for rel_path in triplets:
        f0_path, f1_path, f2_path = vimeo_dataset.triplet_image_paths(rel_path)
        frame0 = Image.open(f0_path).convert("L")
        ground_truth = Image.open(f1_path).convert("L")
        frame2 = Image.open(f2_path).convert("L")
        example_id = rel_path.replace("/", "_")
        save_example(OUTPUT_ROOT / "vimeo" / example_id, frame0, ground_truth, frame2, model, interpreter)
    print(f"wrote {len(triplets)} vimeo examples to {OUTPUT_ROOT / 'vimeo'}")


def run_synthetic(model, interpreter, n=SYNTHETIC_COUNT):
    for i in range(n):
        f0, f1, f2 = training.make_triplet(displacement_range=SYNTHETIC_DISPLACEMENT_RANGE)
        frame0 = array_to_gray_image(f0)
        ground_truth = array_to_gray_image(f1)
        frame2 = array_to_gray_image(f2)
        save_example(OUTPUT_ROOT / "synthetic" / f"{i:03d}", frame0, ground_truth, frame2, model, interpreter)
    print(f"wrote {n} synthetic examples to {OUTPUT_ROOT / 'synthetic'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate interpolated frames for every technique")
    parser.add_argument("--source", choices=["vimeo", "synthetic", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    if not training.TFLITE_MODEL_PATH.exists():
        raise RuntimeError(f"{training.TFLITE_MODEL_PATH} not found — run export_tflite.py first")
    model = load_model()
    interpreter = load_tflite_interpreter()

    if args.source in ("vimeo", "both"):
        run_vimeo(model, interpreter)
    if args.source in ("synthetic", "both"):
        run_synthetic(model, interpreter)


if __name__ == "__main__":
    main()
