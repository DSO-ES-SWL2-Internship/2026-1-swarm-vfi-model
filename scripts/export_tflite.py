# Exports the trained toy U-Net to a fully INT8-quantized TFLite model, for
# deployment on the i.MX8M Plus NPU via NXP's eIQ/VX delegate. Full integer
# quantization (not just weights) is required — the NPU silently falls back
# to CPU for float ops, so both input and output tensors are quantized too.

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import interpolate
import training
import vimeo_dataset

REP_DATASET_SIZE = 100  # samples used to calibrate activation quantization ranges

def main():
  convert()
  fp32_size = training.MODEL_PATH.stat().st_size
  int8_size = training.TFLITE_MODEL_PATH.stat().st_size
  print(f"saved {training.TFLITE_MODEL_PATH} ({int8_size / 1024:.0f} KB, "
        f"vs {fp32_size / 1024:.0f} KB fp32 .keras — {fp32_size / int8_size:.1f}x smaller)\n")
  sanity_check()


# Create a randomly-selected representative dataset
def representative_dataset_gen():
  ds = vimeo_dataset.make_dataset("train.txt", batch_size=1, shuffle=True)
  for x, _ in ds.take(REP_DATASET_SIZE):
    yield [x]


# Convert FP32 model to INT8 (for deployment on iMX8)
def convert():
  model = interpolate.load_model()

  # from_concrete_functions with an explicit batch-size-1 TensorSpec (not from_keras_model,
  # which would preserve the model's own dynamic/None batch dimension) -- every real consumer
  # of this model (relay.py, receiver.py, standalone_inference.py) always runs one frame pair
  # at a time, so a dynamic batch was never actually needed, and it breaks NPU delegation: a
  # dynamic batch dim means Conv2DTranspose's output shape has to be computed at runtime via a
  # STACK op combining several scalar tensors, which TIM-VX's graph compiler cannot handle
  # ("Cannot calculate the reshape tensor 1 to 4") -- confirmed via an isolated single-layer
  # test (same op, fixed batch = runs fully on the NPU delegate; dynamic batch = hard crash).
  # Freezing the batch dimension removes that STACK op entirely, since the output shape becomes
  # a build-time constant instead of something computed from the input at graph-run time.
  input_spec = tf.TensorSpec([1] + list(model.inputs[0].shape[1:]), model.inputs[0].dtype)
  concrete_func = tf.function(model).get_concrete_function(input_spec)
  # tf.function(model) alone leaves the model's weights as ReadVariableOp nodes referencing
  # live tf.Variable resource handles rather than frozen constants -- fine inside a normal
  # Keras/eager context, but the bare concrete function handed to the converter doesn't carry
  # that context, so calibration fails ("READ_VARIABLE ... variable != nullptr was not true").
  # Freezing explicitly embeds the actual weight values as constants in the graph instead.
  frozen_func = convert_variables_to_constants_v2(concrete_func)
  converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func], model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8

  tflite_model = converter.convert()

  training.ARTIFACTS_DIR.mkdir(exist_ok=True)
  training.TFLITE_MODEL_PATH.write_bytes(tflite_model)


# Runs one real example to sanity check for proper inference
def sanity_check():
  interpreter = tf.lite.Interpreter(model_path=str(training.TFLITE_MODEL_PATH))
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  print(f"input:  shape={input_details['shape']} dtype={input_details['dtype'].__name__} "
        f"quantization(scale, zero_point)={input_details['quantization']}")
  print(f"output: shape={output_details['shape']} dtype={output_details['dtype'].__name__} "
        f"quantization(scale, zero_point)={output_details['quantization']}")

  # Run one real example through, to confirm output isn't degenerate (all-zero/saturated).
  ds = vimeo_dataset.make_dataset("eval.txt", batch_size=1, shuffle=False)
  x, y = next(iter(ds.take(1)))

  in_scale, in_zero_point = input_details["quantization"]
  x_q = np.round(x.numpy() / in_scale + in_zero_point).astype(np.int8)
  interpreter.set_tensor(input_details["index"], x_q)
  interpreter.invoke()
  out_q = interpreter.get_tensor(output_details["index"])

  out_scale, out_zero_point = output_details["quantization"]
  out_float = (out_q.astype(np.float32) - out_zero_point) * out_scale

  print(f"sample output range: [{out_float.min():.4f}, {out_float.max():.4f}] "
        f"(ground truth range: [{y.numpy().min():.4f}, {y.numpy().max():.4f}])")


if __name__ == "__main__":
  main()
