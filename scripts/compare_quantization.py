# Compares interpolation quality of the per-channel INT8 model (previous deployment), the per-tensor
# INT8 model (deployed; 1.9x faster on the NPU) and the original fp32 Keras model, on the same Vimeo eval triplets.
# Reports PSNR/SSIM against the true middle frame, plus each INT8 model's PSNR against fp32 (pure
# quantisation error). Runs the TFLite models on the CPU interpreter; the NPU executes the same
# integer arithmetic, so results should match closely (not bit-exact: rounding may differ).
# Usage (on the VM): python3 compare_quantization.py [--samples 300]

import argparse
import statistics

import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import interpolate
import training
import vimeo_dataset

MODELS = {
	"int8 per-channel": training.ARTIFACTS_DIR / "toy_unet_int8_perchannel.tflite",  # previous deployment
	"int8 per-tensor (deployed)": training.TFLITE_MODEL_PATH,  # export_tflite.py with PER_TENSOR = True
}

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=300)
args = parser.parse_args()


def tflite_runner(path):
	interp = tf.lite.Interpreter(model_path=str(path))
	interp.allocate_tensors()
	inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
	in_scale, in_zp = inp["quantization"]
	out_scale, out_zp = out["quantization"]

	def run(x):
		interp.set_tensor(inp["index"], np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8))
		interp.invoke()
		return (interp.get_tensor(out["index"]).astype(np.float32) - out_zp) * out_scale
	return run


runners = {name: tflite_runner(path) for name, path in MODELS.items()}
fp32 = interpolate.load_model()
scores = {name: {"psnr": [], "ssim": [], "psnr_vs_fp32": []} for name in ["fp32 (reference)", *runners]}

for i, (x, y) in enumerate(vimeo_dataset.make_dataset("eval.txt", batch_size=1, shuffle=False).take(args.samples)):
	x, gt = x.numpy(), y.numpy()[0, ..., 0]
	ref = np.clip(fp32(x, training=False).numpy()[0, ..., 0], 0, 1)
	preds = {"fp32 (reference)": ref, **{n: np.clip(r(x)[0, ..., 0], 0, 1) for n, r in runners.items()}}
	for name, pred in preds.items():
		scores[name]["psnr"].append(psnr(gt, pred, data_range=1.0))
		scores[name]["ssim"].append(ssim(gt, pred, data_range=1.0))
		if name != "fp32 (reference)":
			scores[name]["psnr_vs_fp32"].append(psnr(ref, pred, data_range=1.0))
	if (i + 1) % 50 == 0:
		print(f"  {i + 1} samples...", flush=True)

n = len(scores["fp32 (reference)"]["psnr"])
print(f"\n{n} Vimeo eval triplets. PSNR/SSIM vs the true middle frame (higher is better).")
print(f"{'model':<30} {'PSNR dB':>9} {'SSIM':>8} {'PSNR vs fp32':>13}")
for name, s in scores.items():
	vs = f"{statistics.mean(s['psnr_vs_fp32']):13.2f}" if s["psnr_vs_fp32"] else f"{'-':>13}"
	print(f"{name:<30} {statistics.mean(s['psnr']):9.2f} {statistics.mean(s['ssim']):8.4f} {vs}")

# Paired difference on the same triplets: what switching to per-tensor costs, sample by sample.
pc, pt = scores["int8 per-channel"]["psnr"], scores["int8 per-tensor (deployed)"]["psnr"]
diff = [b - a for a, b in zip(pc, pt)]
print(f"\nper-tensor minus per-channel PSNR: mean {statistics.mean(diff):+.3f} dB, "
      f"sd {statistics.stdev(diff):.3f}, per-tensor better on {sum(d > 0 for d in diff)}/{n} samples")
