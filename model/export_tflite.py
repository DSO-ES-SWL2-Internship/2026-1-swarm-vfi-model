"""Exports the trained VFI U-Net as a fully INT8 TFLite model for the iMX8M Plus NPU (NXP's VX delegate).

  * Full integer, input and output too: the NPU silently falls back to the CPU for float operations.
  * One weight scale per layer (per-tensor), not TF's default one per output channel: the NPU is optimised for it
    (NXP UG10166 sec. 8.1.5).
  * Batch fixed at 1: a dynamic batch makes the transposed convolutions compute their output shape with a STACK op, which
    TIM-VX cannot compile (delegation fails).
  * Calibrated on 100 real frame pairs (Blender films, video range). Two sentinel pixels, 0 and 1, keep the input range at
    0-1, so the input scale is exactly 1/255: relay.py feeds uint8 pixels with a zero-point shift alone and checks this.
    The output scale comes from the calibration (the relay rescales through a lookup table).

Usage (on the VM): python3 export_tflite.py                    -> artifacts/vfi_unet.tflite
                   python3 export_tflite.py --per-channel --out artifacts/experiments/per_channel.tflite
Then on the iMX8: benchmark_model --graph=vfi_unet.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import data
import evaluate
import model as vfi

CALIBRATION_PAIRS = 100


def calibration(n=CALIBRATION_PAIRS, seed=0):
	rng = np.random.default_rng(seed)
	pool = data.windows("train")
	pairs = []
	for i in rng.choice(len(pool), n, replace=False):
		f0, _, f2 = data.blender_triplet(*pool[i])
		pairs.append(np.concatenate([f0, f2], axis=-1)[None])
	pairs[0][0, 0, 0, :], pairs[0][0, 0, 1, :] = 0.0, 1.0  # input range 0-1: scale 1/255
	return pairs


def representative_dataset_gen():
	"""Calibration samples in the form TFLite's converter wants (also used by experiments/gpu_layer_test.py)."""
	for pair in calibration():
		yield [pair]


def convert(model, per_tensor=True, representative=representative_dataset_gen):
	"""The INT8 TFLite flatbuffer (bytes) of a Keras model of input shape (1, H, W, 2)."""
	spec = tf.TensorSpec([1] + list(model.inputs[0].shape[1:]), model.inputs[0].dtype)
	frozen = convert_variables_to_constants_v2(tf.function(model).get_concrete_function(spec))
	converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = representative
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.int8
	converter.inference_output_type = tf.int8
	converter._experimental_disable_per_channel = per_tensor
	return converter.convert()


def check(model, path):
	"""The INT8 file on the CPU interpreter against the float model, on the held-out sets (40 clips each: it is slow)."""
	interp = tf.lite.Interpreter(model_path=str(path))
	interp.allocate_tensors()
	inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
	(in_scale, in_zp), (out_scale, out_zp) = inp["quantization"], out["quantization"]
	print(f"input scale x 255 = {in_scale * 255:.4f} (must be 1), output scale {out_scale:.5f}, zero point {out_zp}")

	def run_int8(x):
		frames = []
		for frame in x:
			interp.set_tensor(inp["index"], np.clip(np.round(frame[None] / in_scale + in_zp), -128, 127).astype(np.int8))
			interp.invoke()
			frames.append((interp.get_tensor(out["index"]).astype(np.float32) - out_zp) * out_scale)
		return np.concatenate(frames)

	sets = {k: (x[:40], y[:40]) for k, (x, y) in data.test_sets().items()}
	for label, predict in (("float", lambda x: model.predict(x, verbose=0)), ("int8", run_int8)):
		print(evaluate.summary(label, evaluate.psnr_ssim(predict, sets)))


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--model", type=Path, default=vfi.MODEL_PATH, help="trained .keras model")
	ap.add_argument("--out", type=Path, default=vfi.TFLITE_PATH, help="where to save the .tflite")
	ap.add_argument("--per-channel", action="store_true", help="per-channel weights (the slower NPU path, for comparison)")
	ap.add_argument("--no-check", action="store_true")
	args = ap.parse_args()

	model = vfi.load_model(args.model)

	# Write the converted model
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_bytes(convert(model, per_tensor=not args.per_channel))
	print(f"saved {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")

	# Check model
	if not args.no_check:
		check(model, args.out)


if __name__ == "__main__":
	main()
