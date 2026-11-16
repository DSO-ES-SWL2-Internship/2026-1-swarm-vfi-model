# Builds a training set with more motion than Vimeo-90K's triplets, from the open Blender films in
# datasets/blender/ (Big Buck Bunny, Elephants Dream, Tears of Steel; CC BY, (c) Blender Foundation).
#
# Why: Vimeo-90K's triplets (frames t, t+1, t+2) have little motion, so a model trained on them blurs when a
# gap is wider. The relay's burst recovery fills a gap of 4 frames by halving: the midpoint of frames 4 apart,
# then of frames 2 apart. So the useful example is (frame t, frame t+2g) -> frame t+g, for g = 1 and 2.
#
# 1. Each film is decoded once (ffmpeg) to grey 448x256 frames in VIDEO range (16-235, what the relay's decoded luma
#    plane holds), stored as one uint8 file, datasets/blender/frames/<film>.u8 of shape (N, 256, 448).
# 2. Candidate windows (t, g) are drawn at random; a window is kept when its motion is in range: the mean absolute
#    difference between the outer frames, as a fraction of full scale, must be above MIN_MOTION (not still) and below
#    MAX_MOTION (not a scene cut). Each of the three films has its last TEST_FRACTION of frames kept apart.
# 3. datasets/blender/windows_<split>.npz holds the kept windows: film, t, g.
#
# Usage: python3 blender_triplets.py   (a few minutes: decodes about 45,000 frames of video once)
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

H, W = 256, 448
ROOT = Path(__file__).resolve().parent.parent
BLENDER = ROOT / "datasets" / "blender"
FRAMES = BLENDER / "frames"
FILMS = {
	"bbb": BLENDER / "bbb" / "big_buck_bunny_480p_h264.mov",
	"ed": BLENDER / "ed" / "elephantsdream-480-h264-st-aac.mov",
	"tos": BLENDER / "tos" / "tears_of_steel_720p.mov",
}
TEST_FRACTION = 0.15  # the last 15% of each film is only used for testing
MIN_MOTION, MAX_MOTION = 0.004, 0.12
FPS = 24  # Elephants Dream is 23.4 fps: resampled, so every film is a 24 fps stream like the demo


def decode(name):
	"""Decodes one film to FRAMES/<name>.u8; returns the number of frames."""
	out = FRAMES / f"{name}.u8"
	FRAMES.mkdir(parents=True, exist_ok=True)
	# in_range=tv/out_range=tv: keep the luma values as they are (16-235) rather than stretching to 0-255.
	cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(FILMS[name]), "-an",
	       "-vf", f"fps={FPS},scale={W}:{H}:flags=area:in_range=tv:out_range=tv,format=gray",
	       "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
	size = H * W
	count = 0
	with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc, open(out, "wb") as f:
		while True:
			chunk = proc.stdout.read(size * 64)
			if not chunk:
				break
			f.write(chunk)
			count += len(chunk) // size
	if proc.returncode:
		sys.exit(f"ffmpeg failed on {name}")
	return count


def open_frames(name):
	path = FRAMES / f"{name}.u8"
	n = path.stat().st_size // (H * W)
	return np.memmap(path, dtype=np.uint8, mode="r", shape=(n, H, W))


def motion(frames, t, g):
	a, b = frames[t].astype(np.int16), frames[t + 2 * g].astype(np.int16)
	return np.abs(a - b).mean() / 255


def build(per_film_per_gap, seed):
	rng = np.random.default_rng(seed)
	windows = {"train": [], "test": []}
	for film_id, name in enumerate(FILMS):
		if not (FRAMES / f"{name}.u8").exists():
			print(f"decoding {name} ...", flush=True)
			print(f"  {decode(name)} frames", flush=True)
		frames = open_frames(name)
		n = len(frames)
		split_at = int(n * (1 - TEST_FRACTION))
		for g in (1, 2):
			for split, lo, hi in (("train", 0, split_at - 2 * g), ("test", split_at, n - 2 * g)):
				quota = per_film_per_gap if split == "train" else per_film_per_gap // 6
				got, tries = 0, 0
				while got < quota and tries < quota * 40:
					t = int(rng.integers(lo, hi))
					tries += 1
					m = motion(frames, t, g)
					if MIN_MOTION < m < MAX_MOTION:
						windows[split].append((film_id, t, g))
						got += 1
				print(f"{name} gap {g} {split}: {got} windows ({tries} tried)", flush=True)
	for split, w in windows.items():
		arr = np.array(w, dtype=np.int32)
		np.savez(BLENDER / f"windows_{split}.npz", film=arr[:, 0], t=arr[:, 1], g=arr[:, 2], films=np.array(list(FILMS)))
		print(f"{split}: {len(arr)} windows")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--per-film-per-gap", type=int, default=2000)
	ap.add_argument("--seed", type=int, default=0)
	build(ap.parse_args().per_film_per_gap, ap.parse_args().seed)
