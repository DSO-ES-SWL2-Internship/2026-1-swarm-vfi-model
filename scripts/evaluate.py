"""Scores VFI models on the held-out test sets: PSNR, SSIM and LPIPS, beside the two trivial baselines.

  frame hold    the frame before, copied: what a stutter looks like
  linear blend  the average of the two frames: what the model must beat

Test sets (data.py), none of them trained on: vimeo40 (40 Vimeo-90K triplets), sintel_g1 / sintel_g2 (120 clips each from
the Sintel trailer: midpoint of frames 2 / 4 apart, the burst demo's two steps). Scores are on the full-range scale
(0-255 grey). PSNR and SSIM are higher-is-better; LPIPS (a perceptual distance from a pretrained network, which
punishes the blur that PSNR forgives) is lower-is-better.

Usage (on the VM; --no-lpips if torch is not installed):
  python3 evaluate.py                        the deployed model (artifacts/vfi_unet.keras) and the baselines
  python3 evaluate.py a.keras b.keras        these models instead
"""
import argparse

lpips = None
if __name__ == "__main__":
	try:  # torch (LPIPS) must be imported BEFORE TensorFlow: the other way round it crashes (segfault) in this environment
		import lpips
	except ImportError:
		pass

import numpy as np
import tensorflow as tf

import data
import model as vfi


def baselines():
	return {"frame hold": lambda x: x[..., 0:1], "linear blend": lambda x: x.mean(-1, keepdims=True)}


def predictions(predict, x, batch=8):
	return np.concatenate([predict(x[i:i + batch]) for i in range(0, len(x), batch)])


def quality(pred, truth):
	"""(mean PSNR in dB, mean SSIM) of predicted frames against the true ones, both in video range."""
	p, t = tf.constant(data.to_full_range(pred), tf.float32), tf.constant(data.to_full_range(truth), tf.float32)
	return float(tf.reduce_mean(tf.image.psnr(t, p, 1.0))), float(tf.reduce_mean(tf.image.ssim(t, p, 1.0)))


def psnr_ssim(predict, sets):
	"""{set: (psnr dB, ssim)} for predict(x) -> frames in video range."""
	return {name: quality(predictions(predict, x), y) for name, (x, y) in sets.items()}


def lpips_scores(predict, sets, distance=None):
	"""{set: LPIPS} (AlexNet, the paper's default). Grey frames are repeated into 3 channels, scaled to -1..1.
	Only available when this file is run as a script (see the import order above)."""
	if lpips is None:
		raise RuntimeError("LPIPS needs torch and lpips installed, and evaluate.py run as a script")
	import torch
	distance = distance or lpips.LPIPS(net="alex", verbose=False)
	out = {}
	for name, (x, y) in sets.items():
		total = 0.0
		with torch.no_grad():
			for p, t in zip(data.to_full_range(predictions(predict, x)), data.to_full_range(y)):
				a, b = (torch.from_numpy(v.transpose(2, 0, 1)[None]).float().repeat(1, 3, 1, 1) * 2 - 1 for v in (p, t))
				total += distance(a, b).item()
		out[name] = total / len(x)
	return out


def summary(label, quality, perceptual=None):
	parts = [f"{k} {p:.2f} dB {s:.4f}" + (f" LPIPS {perceptual[k]:.4f}" if perceptual else "") for k, (p, s) in quality.items()]
	return f"{label:28s} " + "  ".join(parts)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("models", nargs="*", help="trained .keras files (default: artifacts/vfi_unet.keras)")
	ap.add_argument("--no-lpips", action="store_true")
	args = ap.parse_args()
	sets = data.test_sets()
	rows = dict(baselines())
	for path in args.models or [vfi.MODEL_PATH]:
		net = vfi.load_model(path)
		rows[str(path).rsplit("/", 1)[-1]] = lambda x, m=net: m.predict(x, verbose=0)
	distance = None
	if not args.no_lpips:
		if lpips is None:
			raise SystemExit("LPIPS needs torch and lpips (pip install lpips); or use --no-lpips")
		distance = lpips.LPIPS(net="alex", verbose=False)
	for label, predict in rows.items():
		print(summary(label, psnr_ssim(predict, sets), None if args.no_lpips else lpips_scores(predict, sets, distance)), flush=True)


if __name__ == "__main__":
	main()
