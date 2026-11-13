# LPIPS (learned perceptual distance, lower is better) for the exploration's models, beside PSNR/SSIM: it compares
# images by the features of a pretrained network, so it punishes blur that PSNR and SSIM forgive.
# Same test sets and the same full-range scale as train_better.py. Grey frames are repeated into 3 channels.
#
# Usage (on the VM): python3 score_lpips.py                                  the baselines only
#                    python3 score_lpips.py a0_base_vimeo a3_res_newdata     only these models, artifacts/better/*.keras
# Appends to artifacts/better/results_lpips.txt.
import sys

import lpips
import numpy as np
import torch
from tensorflow import keras

import interpolate
import train_better as tb
import training
import vimeo_dataset

NET = "alex"  # the LPIPS paper's default network, and the one most frame interpolation papers report


def lpips_score(distance, predict, sets, batch=8):
	out = {}
	for name, (x, y) in sets.items():
		pred = np.concatenate([predict(x[i:i + batch]) for i in range(0, len(x), batch)])
		total = 0.0
		with torch.no_grad():
			for p, t in zip(tb.full(pred), tb.full(y)):  # (H, W, 1) in 0-1
				a, b = (torch.from_numpy(v.transpose(2, 0, 1)[None]).float().repeat(1, 3, 1, 1) * 2 - 1 for v in (p, t))
				total += distance(a, b).item()
		out[name] = total / len(pred)
	return out


def main():
	sets = tb.test_sets()
	distance = lpips.LPIPS(net=NET, verbose=False)
	rows = {}
	if not sys.argv[1:]:  # the baselines only when no model is named (they do not change; scoring them takes minutes)
		rows = {"frame hold": lambda x: x[..., 0:1], "linear blend": lambda x: x.mean(-1, keepdims=True)}
		current = interpolate.load_model(training.MODEL_PATH)
		rows["current model (full range)"] = lambda x: vimeo_dataset.to_video_range(
			current.predict(np.clip(vimeo_dataset.to_full_range(x), 0, 1), verbose=0))
	for name in sys.argv[1:]:
		model = keras.models.load_model(tb.OUT_DIR / f"{name}.keras", compile=False)
		rows[name] = lambda x, m=model: m.predict(x, verbose=0)
	with open(tb.OUT_DIR / "results_lpips.txt", "a") as f:
		for name, fn in rows.items():
			s = lpips_score(distance, fn, sets)
			line = f"{name:28s} " + "  ".join(f"{k} LPIPS {v:.4f}" for k, v in s.items())
			print(line, flush=True)
			f.write(line + "\n")


if __name__ == "__main__":
	main()
