# Exploration: can the toy U-Net's interpolation look better, at a cost the NPU can still afford?
#
# Variants (one flag each):
#   --residual      the output is the average of the two frames plus a learned correction, not a fresh picture: sharp
#                   where nothing moves, the network only has to describe the difference. The average is a fixed 1x1
#                   conv (no channel slicing), so the whole graph stays on the NPU.
#   --width W       channels x W (16/32/64 at W=1). NPU time grows with the arithmetic: 12 ms at W=1.
#   --grad-weight G adds an L1 loss on image gradients, which punishes blur directly.
#   data            blender_triplets.py's windows (more motion; gaps 1 and 2) mixed with Vimeo-90K's train triplets.
#
# Everything is in VIDEO range (16-235 as 0-1), like the relay's decoded luma plane. Scores are computed on the
# full-range scale (the one slide 48's 24.9 dB uses), for every model alike, so they are comparable:
#   vimeo40    the 40 held-out Vimeo triplets (gap 1): the old accuracy test
#   blender_g1 / blender_g2   held-out last 15% of each film: midpoint of frames 2 / 4 apart (the burst demo's two steps)
#
# Usage (on the VM):
#   python3 train_better.py --baselines                       scores of linear blend, frame hold, the current model
#   python3 train_better.py --name res16 --residual --epochs 12
import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import interpolate
import training
import vimeo_dataset

H, W = training.IMG_HEIGHT, training.IMG_WIDTH
BLENDER = training.PROJECT_ROOT / "datasets" / "blender"
OUT_DIR = training.ARTIFACTS_DIR / "better"
RESULTS = OUT_DIR / "results.txt"
_frames = {}


# ----------------------------------------------------------------------------------------------------- data
def film(name):
	if name not in _frames:
		path = BLENDER / "frames" / f"{name}.u8"
		_frames[name] = np.memmap(path, dtype=np.uint8, mode="r", shape=(path.stat().st_size // (H * W), H, W))
	return _frames[name]


def windows(split):
	z = np.load(BLENDER / f"windows_{split}.npz")
	films = [str(f) for f in z["films"]]
	return [(films[f], int(t), int(g)) for f, t, g in zip(z["film"], z["t"], z["g"])]


def blender_triplet(name, t, g):
	f = film(name)
	return tuple(f[i].astype(np.float32)[..., None] / 255 for i in (t, t + g, t + 2 * g))


def vimeo_triplet(rel_path):
	return tuple(vimeo_dataset.to_video_range(f) for f in vimeo_dataset._load_triplet_grayscale(rel_path))


def train_dataset(vimeo_fraction, seed, batch=8):
	rng = np.random.default_rng(seed)
	blender, vimeo = windows("train"), vimeo_dataset.load_manifest("train.txt")

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


def test_sets(per_set=120, seed=1):
	# {name: (X, Y)} in video range, X (n, H, W, 2), Y (n, H, W, 1).
	rng = np.random.default_rng(seed)
	sets = {}
	triplets = [vimeo_triplet(p) for p in vimeo_dataset.load_manifest("eval.txt")]
	sets["vimeo40"] = triplets
	pool = windows("test")
	for g in (1, 2):
		sel = [w for w in pool if w[2] == g]
		sets[f"blender_g{g}"] = [blender_triplet(*sel[i]) for i in rng.choice(len(sel), per_set, replace=False)]
	return {k: (np.stack([np.concatenate([a, c], -1) for a, _, c in v]), np.stack([b for _, b, _ in v]))
	        for k, v in sets.items()}


# ----------------------------------------------------------------------------------------------------- model
def build(width=1.0, residual=False, h=H, w=W):
	c1, c2, c3 = (round(c * width) for c in (16, 32, 64))
	inputs = keras.Input(shape=(h, w, 2))
	e1 = layers.Conv2D(c1, 3, padding="same", activation="relu")(inputs)
	e2 = layers.Conv2D(c2, 3, padding="same", activation="relu")(layers.MaxPooling2D(2)(e1))
	b = layers.Conv2D(c3, 3, padding="same", activation="relu")(layers.MaxPooling2D(2)(e2))
	u2 = layers.Conv2DTranspose(c2, 3, strides=2, padding="same", activation="relu")(b)
	u2 = layers.Conv2D(c2, 3, padding="same", activation="relu")(layers.Concatenate()([u2, e2]))
	u1 = layers.Conv2DTranspose(c1, 3, strides=2, padding="same", activation="relu")(u2)
	u1 = layers.Conv2D(c1, 3, padding="same", activation="relu")(layers.Concatenate()([u1, e1]))
	if not residual:
		return keras.Model(inputs, layers.Conv2D(1, 3, padding="same", activation="sigmoid")(u1), name="unet")
	average = layers.Conv2D(1, 1, use_bias=False, trainable=False, kernel_initializer=keras.initializers.Constant(0.5),
	                        name="average")(inputs)
	delta = layers.Conv2D(1, 3, padding="same", kernel_initializer="zeros", name="delta")(u1)
	out = layers.ReLU(max_value=1.0, name="clip")(layers.Add(name="sum")([average, delta]))
	return keras.Model(inputs, out, name="unet_residual")


def make_loss(grad_weight):
	def loss(y, p):
		l = tf.reduce_mean(tf.sqrt(tf.square(y - p) + 1e-6))  # Charbonnier
		if grad_weight:
			gy, gp = tf.image.image_gradients(y), tf.image.image_gradients(p)
			l += grad_weight * sum(tf.reduce_mean(tf.abs(a - b)) for a, b in zip(gy, gp))
		return l
	return loss


# ----------------------------------------------------------------------------------------------------- scoring
def full(v):
	return np.clip(vimeo_dataset.to_full_range(v), 0, 1)


def score(predict, sets, batch=8):
	# predict(X video range) -> prediction in video range. Scored on the full-range scale. {set: (psnr, ssim)}.
	out = {}
	for name, (x, y) in sets.items():
		pred = np.concatenate([predict(x[i:i + batch]) for i in range(0, len(x), batch)])
		p, t = tf.constant(full(pred), tf.float32), tf.constant(full(y), tf.float32)
		out[name] = (float(tf.reduce_mean(tf.image.psnr(t, p, 1.0))), float(tf.reduce_mean(tf.image.ssim(t, p, 1.0))))
	return out


def fmt(scores):
	return "  ".join(f"{k} {p:.2f} dB {s:.4f}" for k, (p, s) in scores.items())


def baselines(sets):
	rows = {"frame hold": lambda x: x[..., 0:1], "linear blend": lambda x: x.mean(-1, keepdims=True)}
	current = interpolate.load_model(training.MODEL_PATH)  # trained on full range: convert in and out
	rows["current model (full range)"] = lambda x: vimeo_dataset.to_video_range(
		current.predict(np.clip(vimeo_dataset.to_full_range(x), 0, 1), verbose=0))
	return {name: score(fn, sets) for name, fn in rows.items()}


def log(line):
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	print(line, flush=True)
	with open(RESULTS, "a") as f:
		f.write(line + "\n")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--baselines", action="store_true")
	ap.add_argument("--name")
	ap.add_argument("--residual", action="store_true")
	ap.add_argument("--width", type=float, default=1.0)
	ap.add_argument("--grad-weight", type=float, default=0.0)
	ap.add_argument("--vimeo-fraction", type=float, default=0.2)
	ap.add_argument("--epochs", type=int, default=12)
	ap.add_argument("--steps", type=int, default=250, help="batches of 8 per epoch")
	ap.add_argument("--seed", type=int, default=0)
	args = ap.parse_args()
	sets = test_sets()
	if args.baselines:
		for name, s in baselines(sets).items():
			log(f"{name:28s} {fmt(s)}")
		return
	keras.utils.set_random_seed(args.seed)
	model = build(args.width, args.residual)
	total = args.epochs * args.steps
	model.compile(keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, total, alpha=0.05)),
	              loss=make_loss(args.grad_weight))
	params = model.count_params()
	start = time.time()
	model.fit(train_dataset(args.vimeo_fraction, args.seed), steps_per_epoch=args.steps, epochs=args.epochs, verbose=2)
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	model.save(OUT_DIR / f"{args.name}.keras")
	s = score(lambda x: model.predict(x, verbose=0), sets)
	log(f"{args.name:28s} {fmt(s)}  [{params} params, {(time.time() - start) / 60:.0f} min; {vars(args)}]")


if __name__ == "__main__":
	main()
