# Tests ML inference using ONNX runtime
# Provides information and profiling about the use of NPU and other hardware

import argparse
import json
import time
from collections import defaultdict

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
parser.add_argument("--profile", action="store_true",
                     help="enable ONNX Runtime's built-in profiler and summarize which provider "
                          "actually ran each node -- 'active' in get_providers() only means a "
                          "provider is registered for the session, not that it's actually "
                          "claiming any of the graph's real compute nodes.")
args = parser.parse_args()

requested_providers = args.provider or ["VSINPUExecutionProvider", "CPUExecutionProvider"]

# get_available_providers() lists what this onnxruntime BUILD was compiled with, not what's
# active right now -- a plain pip-installed wheel will only ever show CPUExecutionProvider
# here, since the NPU EP only exists in the NXP/Yocto-built onnxruntime.
print(f"Available providers (compiled in): {ort.get_available_providers()}")

session_options = ort.SessionOptions()
if args.profile:
	session_options.enable_profiling = True

session = ort.InferenceSession(args.model, sess_options=session_options, providers=requested_providers)

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

if args.profile:
	# end_profiling() flushes and returns the path to a Chrome-trace-format JSON file. Only
	# "Node"-category events with kernel timing carry a "provider" field in args -- these are
	# the actual per-op executions, as opposed to session-level/allocation events which aren't
	# tied to a specific provider. Summing duration by provider is the real answer to "is the
	# NPU actually doing the work," independent of whether it's merely "active" in the session.
	profile_path = session.end_profiling()
	with open(profile_path) as f:
		events = json.load(f)

	provider_time_us = defaultdict(float)
	provider_count = defaultdict(int)
	for event in events:
		provider = event.get("args", {}).get("provider")
		if event.get("cat") == "Node" and provider:
			provider_time_us[provider] += event.get("dur", 0)
			provider_count[provider] += 1

	total_us = sum(provider_time_us.values())
	print(f"\nPer-provider node breakdown (from {profile_path}):")
	if total_us == 0:
		print("  No provider-tagged Node events found -- this onnxruntime build/version may not "
		      "tag kernel-time events with 'provider' in profiling output.")
	else:
		for provider, time_us in sorted(provider_time_us.items(), key=lambda kv: -kv[1]):
			pct = 100 * time_us / total_us
			print(f"  {provider}: {provider_count[provider]} nodes, "
			      f"{time_us / 1000:.2f} ms total ({pct:.1f}% of node time)")

save_gray(output[0], args.output)
print(f"Saved prediction to {args.output}")

if args.ground_truth:
	ground_truth = load_gray(args.ground_truth, size)
	mae = np.abs(output[0] - ground_truth).mean()
	print(f"Mean absolute error vs ground truth: {mae:.4f}")
