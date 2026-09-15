"""
Loads the real Vimeo-90K triplet dataset from datasets/vimeo90k/, manages the
train/val/eval split manifests under splits/, and builds a tf.data.Dataset
matching training.py's synthetic pipeline's (x, y) shape.
"""

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "vimeo90k"
SPLITS_DIR = PROJECT_ROOT / "splits"

TRAIN_SIZE = 640
VAL_SIZE = 160
EVAL_SIZE = 40
SEED = 42

IMG_HEIGHT = 256  # Vimeo-90K
IMG_WIDTH = 448  # Vimeo-90K


def _scan_triplets():
    """Returns all triplet dirs (as 'group/clip' strings) that contain im1/im2/im3.png."""
    triplets = []
    for group_dir in sorted(DATASET_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        for clip_dir in sorted(group_dir.iterdir()):
            if (clip_dir / "im1.png").exists():
                triplets.append(f"{group_dir.name}/{clip_dir.name}")
    return triplets


def _write_manifest(name, triplets):
    (SPLITS_DIR / name).write_text("\n".join(triplets) + "\n")


def generate_splits():
    """Randomly samples disjoint train/val/eval triplets (fixed seed) and writes manifests."""
    all_triplets = _scan_triplets()
    total_needed = TRAIN_SIZE + VAL_SIZE + EVAL_SIZE
    if len(all_triplets) < total_needed:
        raise ValueError(
            f"Need {total_needed} triplets under {DATASET_ROOT}, found {len(all_triplets)}"
        )

    rng = random.Random(SEED)
    rng.shuffle(all_triplets)

    train = all_triplets[:TRAIN_SIZE]
    val = all_triplets[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    eval_ = all_triplets[TRAIN_SIZE + VAL_SIZE:total_needed]

    SPLITS_DIR.mkdir(exist_ok=True)
    _write_manifest("train.txt", train)
    _write_manifest("val.txt", val)
    _write_manifest("eval.txt", eval_)


def load_manifest(name):
    """Returns the list of 'group/clip' triplet paths in a manifest, generating splits if needed."""
    path = SPLITS_DIR / name
    if not path.exists():
        generate_splits()
    return path.read_text().strip().split("\n")


def triplet_image_paths(rel_path):
    """Returns the (frame0, ground_truth, frame2) file paths for a 'group/clip' triplet."""
    triplet_dir = DATASET_ROOT / rel_path
    return triplet_dir / "im1.png", triplet_dir / "im2.png", triplet_dir / "im3.png"


def _load_gray(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., None]


def _load_triplet_grayscale(rel_path):
    f0_path, f1_path, f2_path = triplet_image_paths(rel_path)
    return _load_gray(f0_path), _load_gray(f1_path), _load_gray(f2_path)


def make_dataset(manifest_name, batch_size=8, shuffle=True):
    """Builds a tf.data.Dataset yielding (x, y) like the synthetic pipeline:
    x = concat(frame0, frame2), shape (H, W, 2); y = frame1 (ground truth), shape (H, W, 1).
    Small enough (640 train triplets) to cache in memory after the first pass."""
    triplets = load_manifest(manifest_name)

    def generator():
        for rel_path in triplets:
            f0, f1, f2 = _load_triplet_grayscale(rel_path)
            x = np.concatenate([f0, f2], axis=-1)
            yield x, f1

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
        ),
    )
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(triplets))
    return ds.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    generate_splits()
    train = load_manifest("train.txt")
    val = load_manifest("val.txt")
    eval_ = load_manifest("eval.txt")
    print(f"train={len(train)} val={len(val)} eval={len(eval_)}")
    overlap = (set(train) & set(val)) | (set(train) & set(eval_)) | (set(val) & set(eval_))
    print(f"overlap between splits: {len(overlap)}")
