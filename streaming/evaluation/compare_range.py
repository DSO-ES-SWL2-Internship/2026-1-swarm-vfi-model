# Accuracy check for the relay's --range lut option against the original --range rgb, before any
# encoding (so codec noise doesn't blur the comparison). Uses the CPU scaler chain, the one whose input
# side --range lut changes (the G2D chain's input is identical in both modes).
#   1. Model input: grey frames from each chain vs the training convention (PIL convert("L") of the
#      decoded RGB frame) -- mean and max grey-level difference.
#   2. Interpolation: the deployed INT8 model (TFLite, CPU interpreter) predicts frame i+1 from frames
#      i and i+2, fed by each chain; PSNR against the PIL-convention middle frame.
#   3. Output side: relay.py's FULL_TO_VIDEO table vs videoconvert's GRAY8 -> RGBx -> I420, all 256 values.
# Runs on the VM (needs gst-launch-1.0, tensorflow, Pillow): python3 compare_range.py [--frames 1251]

import argparse
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent  # machine-learning/
VIDEO = ROOT / "streaming" / "videos" / "sintel_trailer-480p.mp4"
MODEL = ROOT / "artifacts" / "toy_unet_int8.tflite"
H, W = 256, 448
SIZE = f"video/x-raw,width={W},height={H}"
# Same chains as relay.py's SCALERS["cpu"] (rgb) and SCALERS_LUT["cpu"] (lut), decoded on the CPU.
CHAINS = {
	"rgb": f"videoscale ! {SIZE} ! videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=GRAY8",
	"lut": f"videoscale ! {SIZE} ! videoconvert ! video/x-raw,format=GRAY8",
	"reference": f"videoscale ! {SIZE} ! videoconvert ! video/x-raw,format=RGB",
}
VIDEO_TO_FULL = np.clip(np.round((np.arange(256) - 16) * 255 / 219), 0, 255).astype(np.uint8)  # as in relay.py
FULL_TO_VIDEO = np.round(16 + np.arange(256) * 219 / 255).astype(np.uint8)


def gst_to_raw(source, chain, out_path):
	subprocess.run(["gst-launch-1.0", "-q", *source.split(), "!", *chain.split(), "!", "filesink",
	                f"location={out_path}"], check=True)


def psnr(a, b):
	mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
	return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=1251, help="use the first N frames of the video")
args = parser.parse_args()

with tempfile.TemporaryDirectory() as tmp:
	tmp = Path(tmp)
	source = f"filesrc location={VIDEO} ! qtdemux ! h264parse ! avdec_h264"
	for name, chain in CHAINS.items():
		gst_to_raw(source, chain, tmp / f"{name}.raw")
	grey = {name: np.fromfile(tmp / f"{name}.raw", np.uint8).reshape(-1, H, W)[:args.frames] for name in ("rgb", "lut")}
	grey["lut"] = VIDEO_TO_FULL[grey["lut"]]  # what the relay does before handing frames to the model
	rgb = np.fromfile(tmp / "reference.raw", np.uint8).reshape(-1, H, W, 3)[:args.frames]
	reference = np.stack([np.asarray(Image.fromarray(f).convert("L")) for f in rgb])  # training convention

	# Output side: videoconvert's GRAY8 -> RGBx -> I420 on a ramp of all 256 grey values.
	ramp = np.tile(np.arange(256, dtype=np.uint8), H * W // 256).reshape(H, W)
	ramp.tofile(tmp / "ramp.raw")
	gst_to_raw(f"filesrc location={tmp / 'ramp.raw'} blocksize={H * W} ! rawvideoparse format=gray8 width={W} height={H}",
	           "videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=I420", tmp / "ramp_i420.raw")
	videoconvert_table = np.fromfile(tmp / "ramp_i420.raw", np.uint8)[:256]

n = len(reference)
print(f"{VIDEO.name}: first {n} frames, CPU decode and scale to {W}x{H}.\n")
print("1. Model input vs training convention (PIL convert('L')), grey levels:")
for name in ("rgb", "lut"):
	diff = np.abs(grey[name].astype(np.int16) - reference)
	print(f"   {name}: mean |diff| {diff.mean():.2f}, 99th pct {np.percentile(diff, 99):.0f}, max {diff.max()}, "
	      f"PSNR {psnr(grey[name], reference):.1f} dB")

interpreter = tf.lite.Interpreter(model_path=str(MODEL))
interpreter.allocate_tensors()
inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def interpolate(f0, f2):
	# Same arithmetic as relay.py's ToyUnetBlenderTFLite.infer (scales are 1/255 and 1/256).
	x = np.stack([f0.astype(np.int16) + inp["quantization"][1], f2.astype(np.int16) + inp["quantization"][1]], -1)
	interpreter.set_tensor(inp["index"], x[None].astype(np.int8))
	interpreter.invoke()
	return (interpreter.get_tensor(out["index"])[0, ..., 0].astype(np.int16) - out["quantization"][1]).astype(np.uint8)


scores = {"rgb": [], "lut": []}
for i in range(0, n - 2, 2):
	for name in scores:
		scores[name].append(psnr(interpolate(grey[name][i], grey[name][i + 2]), reference[i + 1]))
paired = [b - a for a, b in zip(scores["rgb"], scores["lut"])]
print(f"\n2. Interpolation, {len(paired)} triplets (predict frame i+1 from i and i+2), PSNR vs reference middle frame:")
for name, s in scores.items():
	print(f"   {name}: {statistics.mean(s):.2f} dB")
print(f"   lut minus rgb: mean {statistics.mean(paired):+.3f} dB, sd {statistics.stdev(paired):.3f}, "
      f"lut better on {sum(d > 0 for d in paired)}/{len(paired)}")

table_diff = np.abs(FULL_TO_VIDEO.astype(np.int16) - videoconvert_table)
print(f"\n3. Output side, full -> video range over all 256 grey values: FULL_TO_VIDEO vs videoconvert differ by "
      f"max {table_diff.max()} level(s), on {np.count_nonzero(table_diff)}/256 values")
