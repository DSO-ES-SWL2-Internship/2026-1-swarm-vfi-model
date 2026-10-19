# Exports float variants of the toy U-Net and of its transposed-convolution layer, to test whether the
# GPU runs float models faster than INT8 ones. NXP's ML guide (UG10166 sec 8.1.1) says the i.MX8 GPU is
# optimised for fixed point and half-precision float, so FP16 may suit it where INT8 (9.5 s/inference)
# did not. The NPU is fixed point only, so it should not benefit.
#   fp32: plain float model.
#   fp16: TFLite float16 quantisation -- weights stored as float16 (half the size); TFLite inserts
#         DEQUANTIZE ops back to float32, and a delegate may run the arithmetic in float16 if it chooses.
# Benchmark on the board with benchmarks/benchmark_backends.sh; check the log for how many nodes each
# delegate claimed, since float ops it cannot run fall back to the CPU.
# Usage (on the VM): python3 export_float_variants.py   -> artifacts/float_variants/*.tflite

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import interpolate
import training

OUT_DIR = training.ARTIFACTS_DIR / "float_variants"
H, W = training.IMG_HEIGHT, training.IMG_WIDTH  # 256, 448


def conv_t_layer():
	# The U-Net's first transposed convolution on its own, as "convT" in gpu_layer_test.py (not imported
	# from there: importing that script re-exports all of its models).
	x = keras.Input(shape=(H // 2, W // 2, 32), batch_size=1)
	return keras.Model(x, layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x), name="convT")


def mobilenet():
	# Same network as the board's mobilenet_v1_1.0_224_quant.tflite, with random weights: they don't change
	# the timing, and this avoids downloading the ImageNet weights.
	return keras.applications.MobileNet(input_shape=(224, 224, 3), alpha=1.0, weights=None)


def convert(model, fp16):
	# Same fixed batch 1 and frozen weights as export_tflite.py (a dynamic batch breaks delegation).
	specs = [tf.TensorSpec([1, *inp.shape[1:]], inp.dtype) for inp in model.inputs]
	concrete = tf.function(model).get_concrete_function(specs if len(specs) > 1 else specs[0])
	frozen = convert_variables_to_constants_v2(concrete)
	converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
	if fp16:
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
		converter.target_spec.supported_types = [tf.float16]
	return converter.convert()


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	for name, model in (("toy_unet", interpolate.load_model()), ("convT", conv_t_layer()), ("mobilenet_v1", mobilenet())):
		for suffix, fp16 in (("fp32", False), ("fp16", True)):
			path = OUT_DIR / f"{name}_{suffix}.tflite"
			path.write_bytes(convert(model, fp16))
			print(f"saved {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
	main()
