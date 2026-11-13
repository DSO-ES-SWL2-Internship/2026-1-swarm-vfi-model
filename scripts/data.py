"""Data for the VFI U-Net: training triplets, the held-out test sets, and the grey-range helpers.

Everything is grey, 448x256, in VIDEO range (16-235 stored as 0-1): what the relay's decoded luma plane holds, so the
model takes the decoder's output and returns frames for the encoder with no range conversion. A triplet is
(frame before, frame between, frame after); the model gets the outer two as 2 channels and predicts the middle one.

Training data (train_dataset):
  Vimeo-90K triplets (datasets/vimeo90k/, 640 in splits/train.txt): frames t, t+1, t+2, little motion.
  Open Blender films (blender_triplets.py): windows (t, t+g, t+2g) for g = 1, 2 with more motion; the burst demo fills a
  gap of 4 frames by halving, so it needs midpoints of frames 2 and 4 apart. Augmentation: flips and time reversal.
Test data (test_sets), never trained on:
  vimeo40   the 40 held-out Vimeo triplets in splits/eval.txt
  sintel_g1 / sintel_g2   120 clips each from the Sintel trailer (the demo's own film, animation like the training
            data but a different film): the midpoint of frames 2 / 4 apart
"""
import random
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from model import IMG_HEIGHT as H, IMG_WIDTH as W, PROJECT_ROOT

DATASETS = PROJECT_ROOT / "datasets"
VIMEO = DATASETS / "vimeo90k"
BLENDER = DATASETS / "blender"
SPLITS = PROJECT_ROOT / "splits"
TRAILER = PROJECT_ROOT / "streaming" / "videos" / "sintel_trailer-480p.mp4"
TRAIN_SIZE, VAL_SIZE, EVAL_SIZE, SPLIT_SEED = 640, 160, 40, 42


# ------------------------------------------------------------------------------------------------------ grey ranges
def to_video_range(full):
	"""Full-range grey (0-1 for 0-255) -> video range (16-235, as 0-1)."""
	return (16 + 219 * full) / 255


def to_full_range(video):
	"""Video-range grey -> full range, clipped to 0-1. Scores are computed on this scale, as in the earlier results."""
	return np.clip((255 * video - 16) / 219, 0, 1)


# ------------------------------------------------------------------------------------------------------ Vimeo-90K
def load_manifest(name):
	"""The 'group/clip' triplets of splits/<name> (train.txt, val.txt, eval.txt); the splits are made on first use:
	a fixed-seed random sample of disjoint triplets."""
	path = SPLITS / name
	if not path.exists():
		triplets = sorted(f"{g.name}/{c.name}" for g in VIMEO.iterdir() if g.is_dir() for c in g.iterdir()
		                  if (c / "im1.png").exists())
		if len(triplets) < TRAIN_SIZE + VAL_SIZE + EVAL_SIZE:
			raise ValueError(f"need {TRAIN_SIZE + VAL_SIZE + EVAL_SIZE} triplets under {VIMEO}, found {len(triplets)}")
		random.Random(SPLIT_SEED).shuffle(triplets)
		SPLITS.mkdir(exist_ok=True)
		for split, part in (("train.txt", triplets[:TRAIN_SIZE]), ("val.txt", triplets[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]),
		                    ("eval.txt", triplets[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + EVAL_SIZE])):
			(SPLITS / split).write_text("\n".join(part) + "\n")
	return path.read_text().strip().split("\n")


def vimeo_triplet(rel_path):
	"""(frame0, middle, frame2), each (H, W, 1) float32 in video range."""
	return tuple(to_video_range(np.asarray(Image.open(VIMEO / rel_path / f"im{i}.png").convert("L"), np.float32)[..., None] / 255)
	             for i in (1, 2, 3))


# ------------------------------------------------------------------------------------------------------ Blender films
_films = {}


def film(name):
	# the decoded frames of one film (blender_triplets.py): a memory-mapped uint8 array (N, H, W)
	if name not in _films:
		path = BLENDER / "frames" / f"{name}.u8"
		_films[name] = np.memmap(path, dtype=np.uint8, mode="r", shape=(path.stat().st_size // (H * W), H, W))
	return _films[name]


def windows(split="train"):
	"""[(film, t, g), ...] from blender_triplets.py: frames t, t+g, t+2g."""
	z = np.load(BLENDER / f"windows_{split}.npz")
	films = [str(f) for f in z["films"]]
	return [(films[f], int(t), int(g)) for f, t, g in zip(z["film"], z["t"], z["g"])]


def blender_triplet(name, t, g):
	f = film(name)
	return tuple(f[i].astype(np.float32)[..., None] / 255 for i in (t, t + g, t + 2 * g))


# ------------------------------------------------------------------------------------------------------ training data
def train_dataset(vimeo_fraction=0.2, seed=0, batch=8):
	"""An endless tf.data.Dataset of (x, y): x (H, W, 2) the outer frames, y (H, W, 1) the middle one. Each example is a
	Vimeo triplet with probability vimeo_fraction, else a Blender window; randomly time-reversed and flipped."""
	rng = np.random.default_rng(seed)
	blender, vimeo = windows("train"), load_manifest("train.txt")

	def gen():
		while True:
			if rng.random() < vimeo_fraction:
				f0, f1, f2 = vimeo_triplet(vimeo[rng.integers(len(vimeo))])
			else:
				f0, f1, f2 = blender_triplet(*blender[rng.integers(len(blender))])
			if rng.random() < 0.5:  # time reversed: the midpoint is the same
				f0, f2 = f2, f0
			if rng.random() < 0.5:
				f0, f1, f2 = f0[:, ::-1], f1[:, ::-1], f2[:, ::-1]
			if rng.random() < 0.5:
				f0, f1, f2 = f0[::-1], f1[::-1], f2[::-1]
			yield np.concatenate([f0, f2], axis=-1), f1

	spec = (tf.TensorSpec((H, W, 2), tf.float32), tf.TensorSpec((H, W, 1), tf.float32))
	return tf.data.Dataset.from_generator(gen, output_signature=spec).batch(batch).prefetch(4)


# ------------------------------------------------------------------------------------------------------ test data
def vimeo_eval_set():
	"""(x, y): the 40 held-out Vimeo triplets, x (40, H, W, 2), y (40, H, W, 1), video range."""
	triplets = [vimeo_triplet(p) for p in load_manifest("eval.txt")]
	return np.stack([np.concatenate([a, c], -1) for a, _, c in triplets]), np.stack([b for _, b, _ in triplets])


def sintel_sets(per_gap=120, seed=3):
	"""{'sintel_g1': (x, y), 'sintel_g2': (x, y)}: clips of the Sintel trailer with moderate motion (the mean difference
	of the outer frames between 0.4% and 12%: not still, not a cut), the midpoint of frames 2 / 4 apart."""
	raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(TRAILER), "-vf",
	                      f"fps=24,scale={W}:{H}:flags=area:in_range=tv:out_range=tv,format=gray", "-f", "rawvideo", "-"],
	                     capture_output=True, check=True).stdout
	frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
	rng, sets = np.random.default_rng(seed), {}
	for g in (1, 2):
		chosen = []
		for t in rng.permutation(len(frames) - 2 * g):
			if 0.004 < np.abs(frames[t].astype(int) - frames[t + 2 * g].astype(int)).mean() / 255 < 0.12:
				chosen.append(t)
			if len(chosen) == per_gap:
				break
		x = np.stack([np.stack([frames[t], frames[t + 2 * g]], -1) for t in chosen]).astype(np.float32) / 255
		y = np.stack([frames[t + g] for t in chosen]).astype(np.float32)[..., None] / 255
		sets[f"sintel_g{g}"] = (x, y)
	return sets


def test_sets():
	"""{name: (x, y)}: vimeo40, sintel_g1, sintel_g2."""
	return {"vimeo40": vimeo_eval_set(), **sintel_sets()}
