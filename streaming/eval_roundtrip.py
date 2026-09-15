# Compares two videos frame-by-frame with PSNR/SSIM (grayscale, data_range=255 —
# same convention as machine-learning/scripts/eval.py, for numbers that are
# directly comparable to that project's technique comparisons).
#
# Usage: python eval_roundtrip.py videos/sintel_trailer-480p.mp4 videos/output.mp4

import sys

import av
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def decode_gray_frames(path):
  container = av.open(str(path))
  frames = [frame.to_image().convert("L") for frame in container.decode(video=0)]
  container.close()
  return frames


def main():
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <original.mp4> <roundtrip.mp4>")
    sys.exit(1)

  original_frames = decode_gray_frames(sys.argv[1])
  roundtrip_frames = decode_gray_frames(sys.argv[2])

  if len(original_frames) != len(roundtrip_frames):
    print(f"WARNING: frame count mismatch — original={len(original_frames)} "
          f"roundtrip={len(roundtrip_frames)}")

  n = min(len(original_frames), len(roundtrip_frames))
  psnr_values, ssim_values = [], []
  for i in range(n):
    gt_img, pred_img = original_frames[i], roundtrip_frames[i]
    if gt_img.size != pred_img.size:
      # Pipeline downscales (e.g. videoscale to 448x256), so the ground truth
      # is resized down to match before comparing — otherwise there's nothing
      # to compare pixel-for-pixel.
      gt_img = gt_img.resize(pred_img.size, Image.BILINEAR)

    gt, pred = np.array(gt_img), np.array(pred_img)
    psnr_values.append(psnr(gt, pred, data_range=255))
    ssim_values.append(ssim(gt, pred, data_range=255))

  if not psnr_values:
    print("No comparable frames found.")
    sys.exit(1)

  psnr_values = np.array(psnr_values)
  ssim_values = np.array(ssim_values)
  print(f"Compared {len(psnr_values)} frames")
  print(f"PSNR: mean={psnr_values.mean():.2f} min={psnr_values.min():.2f} max={psnr_values.max():.2f}")
  print(f"SSIM: mean={ssim_values.mean():.4f} min={ssim_values.min():.4f} max={ssim_values.max():.4f}")


if __name__ == "__main__":
  main()
