# INT8 per-tensor export of a train_better.py model for the NPU, as export_tflite.py does for the deployed one: batch 1,
# full-integer, calibrated on 100 real frame pairs (Blender, video range) with the two sentinel pixels that keep the
# input scale at 1/255, which relay.py requires. Then checks it against the float model on the held-out sets.
#
# Usage (on the VM): python3 export_better.py a3_res_newdata       -> artifacts/better/a3_res_newdata.tflite
# Then on the iMX8: benchmark_model --graph=a3_res_newdata.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import train_better as tb
import vimeo_dataset


def calibration(n=100, seed=0):
	rng = np.random.default_rng(seed)
	pool = tb.windows("train")
	pairs = []
	for i in rng.choice(len(pool), n, replace=False):
		f0, _, f2 = tb.blender_triplet(*pool[i])
		pairs.append(np.concatenate([f0, f2], axis=-1)[None])
	pairs[0][0, 0, 0, :], pairs[0][0, 0, 1, :] = 0.0, 1.0  # input range 0-1: scale 1/255
	return pairs


def export(name):
	model = keras.models.load_model(tb.OUT_DIR / f"{name}.keras", compile=False)
	pairs = calibration()
	spec = tf.TensorSpec([1, tb.H, tb.W, 2], tf.float32)
	frozen = convert_variables_to_constants_v2(tf.function(model).get_concrete_function(spec))
	converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = lambda: ([p] for p in pairs)
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.int8
	converter.inference_output_type = tf.int8
	converter._experimental_disable_per_channel = True  # per-tensor: the NPU's fast path
	out = tb.OUT_DIR / f"{name}.tflite"
	out.write_bytes(converter.convert())
	return model, out


def check(model, path):
	# INT8 model on the CPU interpreter vs the float model, on the same sets and scale as train_better.py.
	interp = tf.lite.Interpreter(model_path=str(path))
	interp.allocate_tensors()
	inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
	(in_scale, in_zp), (out_scale, out_zp) = inp["quantization"], outp["quantization"]
	print(f"input scale*255 = {in_scale * 255:.4f} (must be 1), output scale = {out_scale:.5f}, zero point {out_zp}")

	def run(x):
		out = []
		for frame in x:
			interp.set_tensor(inp["index"], np.clip(np.round(frame[None] / in_scale + in_zp), -128, 127).astype(np.int8))
			interp.invoke()
			out.append((interp.get_tensor(outp["index"]).astype(np.float32) - out_zp) * out_scale)
		return np.concatenate(out)

	sets = {k: (x[:40], y[:40]) for k, (x, y) in tb.test_sets().items()}  # 40 each: the CPU interpreter is slow
	for label, fn in (("float", lambda x: model.predict(x, verbose=0)), ("int8", run)):
		print(f"{label:6s} {tb.fmt(tb.score(fn, sets))}")


if __name__ == "__main__":
	model, path = export(sys.argv[1])
	print(f"saved {path} ({path.stat().st_size / 1024:.0f} KB)")
	check(model, path)
