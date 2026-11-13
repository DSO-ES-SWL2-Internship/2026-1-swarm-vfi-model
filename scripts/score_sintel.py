# Scores every model (and the baselines) on windows from the Sintel trailer: natural animation that no model was trained
# on, and the demo's own video. Replaces blender_g1/g2 as the main test: those are the last 15% of each film, which is
# mostly scrolling end credits (checked by looking at the frames), so they say little about natural motion.
# sintel_g1/g2: the midpoint of frames 2 / 4 apart, 120 windows each, the same motion filter as blender_triplets.py.
# Usage (on the VM): python3 score_sintel.py [model names ...]      PSNR/SSIM and LPIPS; appends to results_sintel.txt
import subprocess
import sys

import lpips
import numpy as np

import interpolate
import score_lpips
import train_better as tb
import training
import vimeo_dataset
from tensorflow import keras

TRAILER = training.PROJECT_ROOT.parent / "machine-learning" / "streaming" / "videos" / "sintel_trailer-480p.mp4"  # untracked: only in the main repo


def sintel_sets(per_gap=120, seed=3):
	raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(TRAILER), "-vf",
	                      f"fps=24,scale={tb.W}:{tb.H}:flags=area:in_range=tv:out_range=tv,format=gray", "-f", "rawvideo", "-"],
	                     capture_output=True, check=True).stdout
	frames = np.frombuffer(raw, np.uint8).reshape(-1, tb.H, tb.W)
	rng, sets = np.random.default_rng(seed), {}
	for g in (1, 2):
		chosen = []
		for t in rng.permutation(len(frames) - 2 * g):
			m = np.abs(frames[t].astype(int) - frames[t + 2 * g].astype(int)).mean() / 255
			if 0.004 < m < 0.12:
				chosen.append(t)
			if len(chosen) == per_gap:
				break
		x = np.stack([np.stack([frames[t], frames[t + 2 * g]], -1) for t in chosen]).astype(np.float32) / 255
		y = np.stack([frames[t + g] for t in chosen]).astype(np.float32)[..., None] / 255
		sets[f"sintel_g{g}"] = (x, y)
	return sets


if __name__ == "__main__":
	sets = sintel_sets()
	print({k: len(v[0]) for k, v in sets.items()})
	distance = lpips.LPIPS(net="alex", verbose=False)
	rows = {"frame hold": lambda x: x[..., 0:1], "linear blend": lambda x: x.mean(-1, keepdims=True)}
	current = interpolate.load_model(training.MODEL_PATH)
	rows["current model (full range)"] = lambda x: vimeo_dataset.to_video_range(
		current.predict(np.clip(vimeo_dataset.to_full_range(x), 0, 1), verbose=0))
	for name in sys.argv[1:] or ["a0_base_vimeo", "a1_base_newdata", "a2_res_vimeo", "a3_res_newdata"]:
		model = keras.models.load_model(tb.OUT_DIR / f"{name}.keras", compile=False)
		rows[name] = lambda x, m=model: m.predict(x, verbose=0)
	with open(tb.OUT_DIR / "results_sintel.txt", "a") as f:
		for name, fn in rows.items():
			s, l = tb.score(fn, sets), score_lpips.lpips_score(distance, fn, sets)
			line = f"{name:28s} " + "  ".join(f"{k} {s[k][0]:.2f} dB {s[k][1]:.4f} LPIPS {l[k]:.4f}" for k in sets)
			print(line, flush=True)
			f.write(line + "\n")
