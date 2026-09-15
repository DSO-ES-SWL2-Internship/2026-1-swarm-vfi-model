"""
Checks what's actually accelerating TFLite inference of the toy U-Net, rather
than assuming. Establishes a CPU baseline (always works), then honestly
attempts a GPU delegate and reports whether it actually loaded.

Note on scope: TFLite's GPU delegate is not distributed as a prebuilt binary
for Windows desktop the way it is for Android -- a failure to load it here is
expected on a stock `pip install tensorflow` setup, not necessarily a
misconfiguration. There is no generic NPU delegate in TFLite at all; NPU
acceleration (if this machine even has an NPU -- see WINDOWS_SETUP.md step 1)
would require a different runtime entirely (DirectML / ONNX Runtime / a
vendor SDK), not something this script can test.
"""
import time

import numpy as np
import tensorflow as tf

MODEL_PATH = "../machine-learning/artifacts/toy_unet_int8.tflite"
N_RUNS = 50


def time_inference(interpreter, input_detail, output_detail):
	# random INT8 input in valid range -- we only care about timing/whether
	# it runs, not the output values, for this check
	dummy_input = np.random.randint(-128, 127, size=input_detail["shape"], dtype=np.int8)
	interpreter.set_tensor(input_detail["index"], dummy_input)

	# one warm-up run -- first invoke() often includes one-time setup cost
	# that would skew a timing average if counted
	interpreter.invoke()

	start = time.perf_counter()
	for _ in range(N_RUNS):
		interpreter.invoke()
	elapsed = time.perf_counter() - start
	return elapsed / N_RUNS


print("=== CPU baseline ===")
cpu_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
cpu_interpreter.allocate_tensors()
input_detail = cpu_interpreter.get_input_details()[0]
output_detail = cpu_interpreter.get_output_details()[0]
cpu_time = time_inference(cpu_interpreter, input_detail, output_detail)
print(f"mean inference time: {cpu_time * 1000:.3f} ms (avg over {N_RUNS} runs)")
print("(TFLite's default CPU path uses XNNPACK optimizations automatically -- "
      "this IS already an accelerated-vs-naive baseline, not unoptimized CPU code)")

print("\n=== Attempting GPU delegate ===")
try:
	gpu_delegate = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
	gpu_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, experimental_delegates=[gpu_delegate])
	gpu_interpreter.allocate_tensors()
	gpu_time = time_inference(gpu_interpreter, input_detail, output_detail)
	print(f"GPU delegate loaded successfully.")
	print(f"mean inference time: {gpu_time * 1000:.3f} ms (avg over {N_RUNS} runs)")
	print(f"speedup vs CPU: {cpu_time / gpu_time:.2f}x")
except Exception as e:
	print(f"GPU delegate did not load: {e}")
	print("Expected on a stock Windows pip install -- TFLite's GPU delegate isn't "
	      "prebuilt/distributed for Windows desktop the way CPU/XNNPACK is.")

print("\n=== NPU ===")
print("Not tested here -- TFLite has no generic NPU delegate. If this laptop has an "
      "NPU (check WINDOWS_SETUP.md step 1), using it would mean a different runtime "
      "entirely (e.g. exporting to ONNX and using DirectML/Windows ML), not this script.")
