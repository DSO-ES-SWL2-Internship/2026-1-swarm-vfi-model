import argparse
import time

import numpy as np
import onnxruntime as ort
from PIL import Image


def load_gray(path, size):
	img = Image.open(path).convert("L").resize(size)
	return np.asarray(img, dtype=np.float32)[..., None] / 255.0


def save_gray(arr, path):
	img = (arr.squeeze() * 255).clip(0, 255).astype(np.uint8)
	Image.fromarray(img, mode="L").save(path)


parser = argparse.ArgumentParser(description="Standalone ONNX Runtime inference test -- no GStreamer, "
                                              "just model in, frame out. For isolating NPU/model correctness "
                                              "from pipeline correctness.")
parser.add_argument("model", help="path to .onnx model")
parser.add_argument("frame0", help="path to first input frame image")
parser.add_argument("frame2", help="path to second input frame image")
parser.add_argument("--ground-truth", help="optional path to the true middle frame, for an MAE sanity check")
parser.add_argument("--output", default="standalone_output.png", help="where to save the predicted frame")
parser.add_argument("--width", type=int, default=448)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--provider", action="append",
                     help="execution provider to request, in priority order (repeatable flag). "
                          "Default: try the NPU EP first, fall back to CPU.")
args = parser.parse_args()

requested_providers = args.provider or ["VSINPUExecutionProvider", "CPUExecutionProvider"]

# get_available_providers() lists what this onnxruntime BUILD was compiled with, not what's
# active right now -- a plain pip-installed wheel will only ever show CPUExecutionProvider
# here, since the NPU EP only exists in the NXP/Yocto-built onnxruntime.
print(f"Available providers (compiled in): {ort.get_available_providers()}")

session = ort.InferenceSession(args.model, providers=requested_providers)

# InferenceSession does NOT raise if a requested provider fails to load -- it silently
# falls back to the next one in the list. get_providers() after construction is the only
# reliable way to confirm what's actually active, not just what was asked for.
active_providers = session.get_providers()
print(f"Requested (priority order): {requested_providers}")
print(f"Actually active:            {active_providers}")

npu_requested = any("NPU" in p or "VSI" in p for p in requested_providers)
npu_active = any("NPU" in p or "VSI" in p for p in active_providers)
if npu_requested and not npu_active:
	print("WARNING: an NPU provider was requested but is NOT active -- silently fell back "
	      "to CPU. The inference below is NOT running on the NPU.")
elif npu_active:
	print("NPU execution provider is active.")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

size = (args.width, args.height)
frame0 = load_gray(args.frame0, size)
frame2 = load_gray(args.frame2, size)
x = np.concatenate([frame0, frame2], axis=-1)[None, ...]  # (1, H, W, 2), matches receiver.py's ToyUnetBlenderONNX
print(f"Input shape: {x.shape}, dtype: {x.dtype}")

# Warm-up run excluded from timing -- first call often pays one-off graph optimization /
# kernel compilation overhead that a real streaming pipeline wouldn't repeat per frame.
session.run([output_name], {input_name: x})

n_runs = 20
start = time.perf_counter()
for _ in range(n_runs):
	output = session.run([output_name], {input_name: x})[0]
elapsed = time.perf_counter() - start
print(f"Mean inference latency over {n_runs} runs: {elapsed / n_runs * 1000:.2f} ms")
print(f"Output shape: {output.shape}, dtype: {output.dtype}, min/max: {output.min():.3f}/{output.max():.3f}")

save_gray(output[0], args.output)
print(f"Saved prediction to {args.output}")

if args.ground_truth:
	ground_truth = load_gray(args.ground_truth, size)
	mae = np.abs(output[0] - ground_truth).mean()
	print(f"Mean absolute error vs ground truth: {mae:.4f}")
