"""Compares the deployed model in fp32, INT8 per-channel and INT8 per-tensor on the held-out test sets.

Per-tensor weights (one scale per layer) are what the NPU is optimised for: 1.9x faster than TF's default per-channel
(one scale per output channel). This shows what that costs in quality. For each set: PSNR / SSIM against the true
middle frame, PSNR against the fp32 model (the pure quantisation error), and the per-clip difference between the two
INT8 exports. The TFLite files run on the CPU interpreter; the NPU executes the same integer arithmetic, so the results
should match closely (not bit for bit).

Run on the VM after exporting both:
  python3 export_tflite.py                                              artifacts/vfi_unet.tflite (per-tensor)
  python3 export_tflite.py --per-channel --out ../artifacts/experiments/per_channel.tflite --no-check
  python3 compare_quantization.py
Time them on the board with benchmarks/benchmark_backends.sh.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/: model, data, evaluate, export_tflite

import numpy as np
import tensorflow as tf

import data
import evaluate
import model as vfi

PER_CHANNEL = vfi.EXPERIMENTS_DIR / "per_channel.tflite"


def tflite_runner(path):
	interp = tf.lite.Interpreter(model_path=str(path))
	interp.allocate_tensors()
	inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
	(in_scale, in_zp), (out_scale, out_zp) = inp["quantization"], out["quantization"]

	def predict(x):
		frames = []
		for frame in x:
			interp.set_tensor(inp["index"], np.clip(np.round(frame[None] / in_scale + in_zp), -128, 127).astype(np.int8))
			interp.invoke()
			frames.append((interp.get_tensor(out["index"]).astype(np.float32) - out_zp) * out_scale)
		return np.concatenate(frames)
	return predict


def psnr_per_clip(pred, truth):
	mse = ((data.to_full_range(pred) - data.to_full_range(truth)) ** 2).mean(axis=(1, 2, 3))
	return 10 * np.log10(1.0 / np.maximum(mse, 1e-10))


def main():
	for path in (vfi.TFLITE_PATH, PER_CHANNEL):
		if not path.exists():
			raise SystemExit(f"missing {path}: see the usage at the top of this file")
	net = vfi.load_model()
	rows = {"fp32": lambda x: net.predict(x, verbose=0),
	        "int8 per-channel": tflite_runner(PER_CHANNEL),
	        "int8 per-tensor (deployed)": tflite_runner(vfi.TFLITE_PATH)}
	sets = data.test_sets()
	preds = {label: {name: evaluate.predictions(fn, x) for name, (x, _) in sets.items()} for label, fn in rows.items()}
	for label in rows:
		print(evaluate.summary(label, {name: evaluate.quality(preds[label][name], y) for name, (_, y) in sets.items()}))
	print("\nPSNR against fp32 (the quantisation error alone, higher is better) and per-tensor minus per-channel PSNR per clip:")
	for name, (_, y) in sets.items():
		vs = {l: psnr_per_clip(preds[l][name], preds["fp32"][name]).mean() for l in ("int8 per-channel", "int8 per-tensor (deployed)")}
		diff = psnr_per_clip(preds["int8 per-tensor (deployed)"][name], y) - psnr_per_clip(preds["int8 per-channel"][name], y)
		print(f"  {name:10s} vs fp32: per-channel {vs['int8 per-channel']:.2f} dB, per-tensor {vs['int8 per-tensor (deployed)']:.2f} dB;"
		      f" per-tensor minus per-channel {diff.mean():+.3f} dB (sd {diff.std():.3f}, better on {(diff > 0).sum()}/{len(diff)} clips)")


if __name__ == "__main__":
	main()
