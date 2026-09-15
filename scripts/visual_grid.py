# Builds a grid-of-photos visual comparison from interpolate.py's output folders:
# columns = [Frame0, Ground Truth, Frame Hold, Linear Blend, Toy U-Net, Frame2],
# for the first example of a source. Frame0/Frame2 (the model's inputs, not
# predictions) get a light gray mat behind them to set them apart at a glance.

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
COLUMNS = ["frame0", "ground_truth", "frame_hold", "linear_blend", "toy_unet", "toy_unet_int8", "frame2"]
COLUMN_TITLES = [
    "Frame 0 (before)", "Ground Truth", "Frame Hold", "Linear Blend",
    "Toy U-Net (fp32)", "Toy U-Net (INT8)", "Frame 2 (after)",
]
INPUT_COLUMNS = {"frame0", "frame2"}
INPUT_MAT_COLOR = "#e1e0d9"  # faint neutral gray, distinguishes model inputs from predictions/ground truth
MAT_PAD_FRAC = 0.06  # margin around the image, as a fraction of its largest dimension


def build_grid(source, out_path):
  example_dirs = sorted((OUTPUT_ROOT / source).glob("*"))
  if not example_dirs:
    raise RuntimeError(f"No examples found under {OUTPUT_ROOT / source} — run interpolate.py first")
  example_dir = example_dirs[0]

  n_cols = len(COLUMNS)
  fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.6))

  for col, key in enumerate(COLUMNS):
    ax = axes[col]
    img = Image.open(example_dir / f"{key}.png")
    w, h = img.size
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    if key in INPUT_COLUMNS:
      pad = MAT_PAD_FRAC * max(w, h)
      ax.set_facecolor(INPUT_MAT_COLOR)
      ax.set_xlim(-0.5 - pad, w - 0.5 + pad)
      ax.set_ylim(h - 0.5 + pad, -0.5 - pad)  # inverted: image y-origin is top-left

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(COLUMN_TITLES[col], fontsize=9)

  fig.suptitle(f"Visual comparison ({source}): {example_dir.name}")
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)
  print(f"saved {out_path}")


def main():
  ARTIFACTS_DIR.mkdir(exist_ok=True)
  build_grid("vimeo", ARTIFACTS_DIR / "visual_grid_vimeo.png")
  build_grid("synthetic", ARTIFACTS_DIR / "visual_grid_synthetic.png")


if __name__ == "__main__":
  main()
