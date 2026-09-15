# Computes PSNR/SSIM/LPIPS for every technique's outputs 
# and renders a box-and-whisker plot per metric.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TECHNIQUES = ["frame_hold", "linear_blend", "toy_unet", "toy_unet_int8"]
METRICS = ["psnr", "ssim", "lpips"]

# Fixed categorical order — never cycled/reassigned so a technique keeps its color across runs.
TECHNIQUE_COLORS = {
    "frame_hold": "#2a78d6",
    "linear_blend": "#eb6834",
    "toy_unet": "#1baf7a",
    "toy_unet_int8": "#eda100",
}

# PSNR/SSIM: higher = closer to ground truth. LPIPS is a learned *distance*, so lower = more similar.
METRIC_DIRECTION = {"psnr": "higher is better", "ssim": "higher is better", "lpips": "lower is better"}
METRIC_ARROW = {"psnr": "↑", "ssim": "↑", "lpips": "↓"}

def main():
  per_metric, n_examples = collect_results()
  print_summary(per_metric, n_examples)
  plot_box_whisker(per_metric)


def pil_to_lpips_tensor(img):
  # img: grayscale PIL Image (mode L). LPIPS's backbone is RGB-trained, so the
  # single channel is replicated to 3 identical channels only for this call.
  arr = np.array(img).astype(np.float32)
  arr = arr / 255.0 * 2 - 1
  tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
  tensor = tensor.repeat(3, 1, 1)  # (3, H, W)
  return tensor.unsqueeze(0)  # (1, 3, H, W)


def evaluate_example(example_dir, loss_fn):
  ground_truth = Image.open(example_dir / "ground_truth.png").convert("L")
  gt_arr = np.array(ground_truth)
  gt_t = pil_to_lpips_tensor(ground_truth)

  results = {}
  for technique in TECHNIQUES:
    pred_img = Image.open(example_dir / f"{technique}.png").convert("L")
    pred_arr = np.array(pred_img)
    pred_t = pil_to_lpips_tensor(pred_img)

    results[technique] = {
        "psnr": psnr(gt_arr, pred_arr, data_range=255),
        "ssim": ssim(gt_arr, pred_arr, data_range=255),
        "lpips": loss_fn(gt_t, pred_t).item(),
    }
  return results


def collect_results():
  example_dirs = sorted(p for p in OUTPUT_ROOT.glob("*/*") if p.is_dir())
  if not example_dirs:
    raise RuntimeError(f"No examples found under {OUTPUT_ROOT} — run interpolate.py first")

  loss_fn = lpips.LPIPS(net="alex")

  per_metric = {metric: {t: [] for t in TECHNIQUES} for metric in METRICS}
  for example_dir in example_dirs:
    results = evaluate_example(example_dir, loss_fn)
    for technique in TECHNIQUES:
      for metric in METRICS:
        per_metric[metric][technique].append(results[technique][metric])
  return per_metric, len(example_dirs)


def print_summary(per_metric, n_examples):
  print(f"Evaluated {n_examples} examples\n")
  for metric in METRICS:
    print(f"{metric}:")
    for technique in TECHNIQUES:
      values = np.array(per_metric[metric][technique])
      print(f"\t{technique}: mean={values.mean():.4f} std={values.std():.4f}")


def plot_box_whisker(per_metric, out_path=None):
  out_path = out_path or (ARTIFACTS_DIR / "metrics_boxplot.png")
  ARTIFACTS_DIR.mkdir(exist_ok=True)
  fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
  for ax, metric in zip(axes, METRICS):
    data = [per_metric[metric][t] for t in TECHNIQUES]
    bp = ax.boxplot(data, tick_labels=TECHNIQUES, patch_artist=True, medianprops={"color": "#0b0b0b", "linewidth": 2})
    for box, technique in zip(bp["boxes"], TECHNIQUES):
      box.set_facecolor(TECHNIQUE_COLORS[technique])
      box.set_edgecolor("#0b0b0b")
    ax.set_title(f"{metric.upper()} ({METRIC_ARROW[metric]} {METRIC_DIRECTION[metric]})")
    ax.set_ylabel(metric)
  fig.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)
  print(f"\nsaved {out_path}")


if __name__ == "__main__":
  main()
