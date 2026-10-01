# Builds tiny single-layer INT8 TFLite models mirroring the toy U-Net's layers (real shapes, full
# 448x256 resolution), each exported per-channel (TF's default, like toy_unet_int8.tflite) and
# per-tensor, plus the real trained U-Net exported per-tensor. Benchmarked on the board's CPU/NPU/GPU
# (run_gpu_layer_test.sh), they show which layer type or quantisation style makes the GPU so slow
# on our model (9.4 s/inference) when it handles MobileNet at CPU-like speed.
# Usage (on the VM): python3 gpu_layer_test.py   -> artifacts/gpu_layer_test/*.tflite

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import export_tflite
import interpolate
import training

OUT_DIR = training.ARTIFACTS_DIR / "gpu_layer_test"
H, W = training.IMG_HEIGHT, training.IMG_WIDTH  # 256, 448


# Build single layer with name and model
def single_layer(name, input_shapes, build):
	inputs = [keras.Input(shape=s, batch_size=1) for s in input_shapes]
	out = build(*inputs)
	return name, keras.Model(inputs, out, name=name)


LAYER_MODELS = [
	single_layer("conv_in", [(H, W, 2)], lambda x: layers.Conv2D(16, 3, padding="same", activation="relu")(x)),
	single_layer("conv_full", [(H, W, 32)], lambda x: layers.Conv2D(16, 3, padding="same", activation="relu")(x)),
	single_layer("maxpool", [(H, W, 16)], lambda x: layers.MaxPooling2D(pool_size=2)(x)),
	single_layer("convT", [(H // 2, W // 2, 32)],
	             lambda x: layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)),
	single_layer("concat", [(H, W, 16), (H, W, 16)], lambda a, b: layers.Concatenate()([a, b])),
	single_layer("upsample_nearest", [(H // 2, W // 2, 32)], lambda x: layers.UpSampling2D(2)(x)),
]


def convert(model, representative, per_tensor):
	# Same recipe as export_tflite.py: fixed batch 1 (a dynamic batch breaks NPU delegation), frozen
	# weights, full INT8 in and out. per_tensor switches off TF's default per-channel weight scales.
	specs = [tf.TensorSpec([1, *inp.shape[1:]], inp.dtype) for inp in model.inputs]
	# A multi-input Keras model takes its inputs as one list, a single-input model as one tensor.
	concrete = tf.function(model).get_concrete_function(specs if len(specs) > 1 else specs[0])
	frozen = convert_variables_to_constants_v2(concrete)
	converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = representative
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.int8
	converter.inference_output_type = tf.int8
	converter._experimental_disable_per_channel = per_tensor
	return converter.convert()


def random_representative(model):
	# Values in [0, 1] like the real (normalised) frames; weights are random anyway, so the exact
	# data only matters for choosing plausible activation ranges.
	def gen():
		rng = np.random.default_rng(0)
		for _ in range(20):
			yield [rng.random((1, *inp.shape[1:]), dtype=np.float32) for inp in model.inputs]
	return gen


OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, model in LAYER_MODELS:
	for per_tensor in (False, True):
		path = OUT_DIR / f"{name}_{'pertensor' if per_tensor else 'perchannel'}.tflite"
		path.write_bytes(convert(model, random_representative(model), per_tensor))
		print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")

# The real trained model, per-tensor, calibrated on the same Vimeo data as export_tflite.py.
unet = interpolate.load_model()
path = OUT_DIR / "toy_unet_pertensor.tflite"
path.write_bytes(convert(unet, export_tflite.representative_dataset_gen, per_tensor=True))
print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")
