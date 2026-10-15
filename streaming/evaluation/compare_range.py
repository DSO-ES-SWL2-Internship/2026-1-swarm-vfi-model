# Checks that the relay's --range lut option is at least as accurate as the original --range rgb.
#
# Background: H.264 video stores brightness in "video range" (16-235), but the model was trained on
# "full range" grey (0-255, PIL convert("L") of RGB frames). So the relay has to stretch the range before
# the model sees a frame, and squeeze it back afterwards. It has two ways to do this:
#   rgb (default): videoconvert turns YUV into RGB (which also stretches to full range), then RGB into grey.
#   lut:           videoconvert copies the Y (brightness) plane straight to grey, still 16-235, and a lookup
#                  table (VIDEO_TO_FULL) stretches it to 0-255 in numpy. This skips two full-frame conversions.
# Both should give the same grey: PIL's grey is 0.299 R + 0.587 G + 0.114 B, the same weights BT.601 uses
# to build Y, so Y is already the right grey, just squeezed into 16-235.
#
# Reference (what both are scored against): the training convention. ffmpeg converts YUV to RGB
# with the BT.601 formula (the source video's colour standard) and accurate rounding, then PIL's
# convert("L") makes it grey, as vimeo_dataset.py does for the training frames.
#
# Three checks, all before encoding (so codec noise doesn't blur the comparison):
#   1. Model input: how far each method's grey frames are from the reference, in grey levels.
#   2. Interpolation: the deployed INT8 model (TFLite, on the CPU) predicts each middle frame from its
#      neighbours, fed by each method; PSNR of the prediction against the reference middle frame.
#   3. Output side: the reverse table (FULL_TO_VIDEO) vs videoconvert's grey -> RGBx -> I420, for all 256 values.
#
# Only the CPU chain is tested: decoding and scaling are the same for both methods, and the G2D chain
# (which outputs RGB) is not changed by --range lut.
#
# Runs on the VM (needs gst-launch-1.0, ffmpeg, tensorflow, Pillow): python3 compare_range.py [--frames 1251]

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
DECODE = f"filesrc location={VIDEO} ! qtdemux ! h264parse ! avdec_h264"  # CPU decoder, as in the relay

# GStreamer chains after decoding: the same scaling, then each method's colour conversion (relay.py's
# INPUT_CONVERTERS["cpu"] and INPUT_CONVERTERS_LUT["cpu"]). "i420" keeps the colour, for the reference.
CHAINS = {
	"rgb": f"videoscale ! {SIZE} ! videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=GRAY8",
	"lut": f"videoscale ! {SIZE} ! videoconvert ! video/x-raw,format=GRAY8",
	"i420": f"videoscale ! {SIZE} ! videoconvert ! video/x-raw,format=I420",
}
# The relay's lookup tables (copied from relay.py)
VIDEO_TO_FULL = np.clip(np.round((np.arange(256) - 16) * 255 / 219), 0, 255).astype(np.uint8) # Covnersion from Y(UV) to grey
FULL_TO_VIDEO = np.round(16 + np.arange(256) * 219 / 255).astype(np.uint8) # Conversion from grey to Y(UV)


def run_gst(source, chain, out_path):
	# Runs source ! chain and writes the raw frames to out_path.
	subprocess.run(["gst-launch-1.0", "-q", *source.split(), "!", *chain.split(), "!", "filesink",
	                f"location={out_path}"], check=True)


def psnr(a, b):
	mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
	return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def grey_frames(tmp, frames):
	# Each method's grey frames, as the model would receive them.
	grey = {}
	for name in ("rgb", "lut"):
		run_gst(DECODE, CHAINS[name], tmp / f"{name}.raw")
		grey[name] = np.fromfile(tmp / f"{name}.raw", np.uint8).reshape(-1, H, W)[:frames]
	grey["lut"] = VIDEO_TO_FULL[grey["lut"]]  # the relay applies the table before the model
	return grey


def reference_frames(tmp, frames):
	# Same decoding and scaling, kept in colour (I420), then converted the way the training data was.
	# The source is BT.601, video range (avdec_h264 caps: colorimetry=2:4:5:3).
	run_gst(DECODE, CHAINS["i420"], tmp / "i420.raw")
	subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}",
	                "-i", str(tmp / "i420.raw"), "-vf", "scale=in_color_matrix=bt601:in_range=tv:out_range=pc",
	                "-sws_flags", "accurate_rnd+full_chroma_int+bitexact", "-pix_fmt", "rgb24", "-f", "rawvideo",
	                str(tmp / "reference.raw")], check=True)
	rgb = np.fromfile(tmp / "reference.raw", np.uint8).reshape(-1, H, W, 3)[:frames]
	return np.stack([np.asarray(Image.fromarray(f).convert("L")) for f in rgb])


def videoconvert_output_table(tmp):
	# What --range rgb's output converter (grey -> RGBx -> I420) does to each grey value: run a frame
	# holding every value 0-255 through it and read the converted values back from the Y plane.
	ramp = np.tile(np.arange(256, dtype=np.uint8), H * W // 256).reshape(H, W)
	ramp.tofile(tmp / "ramp.raw")
	run_gst(f"filesrc location={tmp / 'ramp.raw'} blocksize={H * W} ! rawvideoparse format=gray8 width={W} height={H}",
	        "videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=I420", tmp / "ramp_i420.raw")
	return np.fromfile(tmp / "ramp_i420.raw", np.uint8)[:256]


class Interpolator:
	# The deployed INT8 model, with the same arithmetic as relay.py's ToyUnetBlenderTFLite.infer: the input
	# and output scales are 1/255 and 1/256, so converting between pixels and INT8 is just a zero-point shift.
	def __init__(self):
		self.interpreter = tf.lite.Interpreter(model_path=str(MODEL))
		self.interpreter.allocate_tensors()
		self.inp = self.interpreter.get_input_details()[0]
		self.out = self.interpreter.get_output_details()[0]

	def __call__(self, before, after):
		zero_in, zero_out = self.inp["quantization"][1], self.out["quantization"][1]
		x = np.stack([before.astype(np.int16) + zero_in, after.astype(np.int16) + zero_in], -1)
		self.interpreter.set_tensor(self.inp["index"], x[None].astype(np.int8))
		self.interpreter.invoke()
		return (self.interpreter.get_tensor(self.out["index"])[0, ..., 0].astype(np.int16) - zero_out).astype(np.uint8)


def check_model_input(grey, reference):
	print("1. Model input vs reference, grey levels:")
	for name, frames in grey.items():
		diff = frames.astype(np.int16) - reference
		print(f"   {name}: mean diff {diff.mean():+.2f}, mean |diff| {np.abs(diff).mean():.2f}, "
		      f"99th pct {np.percentile(np.abs(diff), 99):.0f}, max {np.abs(diff).max()}, PSNR {psnr(frames, reference):.1f} dB")


def check_interpolation(grey, reference):
	# Every other frame is a middle frame: predict frame i from frames i-1 and i+1, as if i were missing.
	interpolate = Interpolator()
	middles = range(1, len(reference) - 1, 2)
	scores = {name: [psnr(interpolate(frames[i - 1], frames[i + 1]), reference[i]) for i in middles]
	          for name, frames in grey.items()}
	gains = [lut - rgb for rgb, lut in zip(scores["rgb"], scores["lut"])]  # per triplet, so scene content cancels out
	print(f"\n2. Interpolation, {len(gains)} triplets (predict frame i+1 from i and i+2), PSNR vs reference middle frame:")
	print(f"   rgb {statistics.mean(scores['rgb']):.2f} dB, lut {statistics.mean(scores['lut']):.2f} dB; "
	      f"lut minus rgb {statistics.mean(gains):+.3f} dB (sd {statistics.stdev(gains):.3f}), "
	      f"lut better on {sum(g > 0 for g in gains)}/{len(gains)}")


def check_output_table(videoconvert_table):
	diff = np.abs(FULL_TO_VIDEO.astype(np.int16) - videoconvert_table)
	print(f"\n3. Output side, full -> video range over all 256 grey values: FULL_TO_VIDEO vs videoconvert differ by "
	      f"max {diff.max()} level(s), on {np.count_nonzero(diff)}/256 values")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--frames", type=int, default=1251, help="use the first N frames of the video")
	args = parser.parse_args()

	with tempfile.TemporaryDirectory() as tmp:
		tmp = Path(tmp)

		# Method comparison (RGB method vs LUT method)
		grey = grey_frames(tmp, args.frames) # Generate grey frames from the two methods
		reference = reference_frames(tmp, args.frames) # Generate the reference grey frames using the ffmpeg method

		# Understanding RGB method
		videoconvert_table = videoconvert_output_table(tmp) # Test the mapping of grey -> RGBx to Y(UV) values

	print(f"{VIDEO.name}: first {len(reference)} frames, CPU decode and scale to {W}x{H}.\n")

	check_model_input(grey, reference) # Before VFI
	check_interpolation(grey, reference) # After VFI
	check_output_table(videoconvert_table) # After VFI AND conversion back to YUV


if __name__ == "__main__":
	main()
