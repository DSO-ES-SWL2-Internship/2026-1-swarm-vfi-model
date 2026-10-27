# Is a U-Net trained on video-range grey (16-235, training.py --range video) as good as the full-range one (0-255,
# deployed when this was written)? With the video-range model the relay needs no range lookups, which cost about
# 3 ms each on the board. It was adopted: relay.py now runs the video-range model only (the full-range relay with
# its lookups is at git tag before-simplify).
#
# Two comparisons on the same held-out Vimeo triplets (eval + val, each once; neither is used for training):
#   fp32      the Keras models, in floating point. The video model's output is mapped back to full range, so both
#             are scored on the same 0-1 scale (scoring in video range would add 20*log10(255/219) = 1.3 dB).
#   pipeline  the INT8 TFLite models the NPU runs, with every step of the relay emulated in uint8:
#             the stream's luma is video range, so the input is round(video-range frame);
#             full model   the old relay's lut mode: VIDEO_TO_FULL lookup -> model -> FULL_TO_VIDEO lookup
#             video model  the relay now: model directly
#             Both outputs are video-range uint8, scored against the video-range ground truth (data range 219).
# Each range is trained with the same seeds, so the run-to-run spread shows how big a difference has to be to
# matter. Paired differences (video minus full, same triplet, same seed) are reported for the pipeline scores.
#
# Usage (on the VM), for each seed S and range R in full, video:
#   python3 training.py --range R --seed S --out ../artifacts/range_test/toy_unet_R_sS.keras
#   python3 export_tflite.py --range R --model ../artifacts/range_test/toy_unet_R_sS.keras \
#       --out ../artifacts/range_test/toy_unet_R_sS_int8.tflite
# then: python3 compare_range.py [--seeds 1 2]

import argparse
import statistics
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import interpolate
import training
import vimeo_dataset

RUN_DIR = training.ARTIFACTS_DIR / "range_test"
# relay.py's lookup tables (--range lut)
VIDEO_TO_FULL = np.clip(np.round((np.arange(256) - 16) * 255 / 219), 0, 255).astype(np.uint8)
FULL_TO_VIDEO = np.round(16 + np.arange(256) * 219 / 255).astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
args = parser.parse_args()


def tflite_uint8(path):
	# Runs an INT8 model the way relay.py does: uint8 pixels in, uint8 pixels out, zero-point shifts only.
	interp = tf.lite.Interpreter(model_path=str(path))
	interp.allocate_tensors()
	inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
	assert abs(inp["quantization"][0] * 255 - 1) < 1e-4, f"{path.name}: input scale is not 1/255"

	def run(f0, f2):
		x = np.stack([f0, f2], axis=-1)[None].astype(np.int16) + inp["quantization"][1]
		interp.set_tensor(inp["index"], x.astype(np.int8))
		interp.invoke()
		return (interp.get_tensor(out["index"])[0, ..., 0].astype(np.int16) - out["quantization"][1]).astype(np.uint8)
	return run


def triplets():
	for name in ("eval.txt", "val.txt"):
		for x, y in vimeo_dataset.make_dataset(name, batch_size=1, shuffle=False, repeat=False):
			yield x.numpy(), y.numpy()[0, ..., 0]


scores = {}  # (kind, range, seed) -> {"psnr": [...], "ssim": [...]}
for seed in args.seeds:
	keras = {r: interpolate.load_model(RUN_DIR / f"toy_unet_{r}_s{seed}.keras") for r in ("full", "video")}
	int8 = {r: tflite_uint8(RUN_DIR / f"toy_unet_{r}_s{seed}_int8.tflite") for r in ("full", "video")}
	for x, gt in triplets():
		# fp32, scored in full range
		preds = {"full": keras["full"](x, training=False).numpy()[0, ..., 0],
		         "video": vimeo_dataset.to_full_range(
			         keras["video"](vimeo_dataset.to_video_range(x), training=False).numpy()[0, ..., 0])}
		for r, pred in preds.items():
			s = scores.setdefault(("fp32", r, seed), {"psnr": [], "ssim": []})
			pred = np.clip(pred, 0, 1)
			s["psnr"].append(psnr(gt, pred, data_range=1.0))
			s["ssim"].append(ssim(gt, pred, data_range=1.0))
		# pipeline: uint8 video-range frames, as the relay gets them from the stream
		f0, f2 = (np.round(255 * vimeo_dataset.to_video_range(x[0, ..., i])).astype(np.uint8) for i in (0, 1))
		gt_v = np.round(255 * vimeo_dataset.to_video_range(gt)).astype(np.uint8)
		preds = {"full": FULL_TO_VIDEO[int8["full"](VIDEO_TO_FULL[f0], VIDEO_TO_FULL[f2])],
		         "video": int8["video"](f0, f2)}
		for r, pred in preds.items():
			s = scores.setdefault(("pipeline", r, seed), {"psnr": [], "ssim": []})
			s["psnr"].append(psnr(gt_v.astype(float), pred.astype(float), data_range=219))
			s["ssim"].append(ssim(gt_v.astype(float), pred.astype(float), data_range=219))
	print(f"seed {seed} done ({len(s['psnr'])} triplets)", flush=True)

n = len(next(iter(scores.values()))["psnr"])
print(f"\n{n} held-out Vimeo triplets (eval + val), mean PSNR / SSIM vs the true middle frame")
print(f"{'':10} {'range':6} " + "".join(f"{'seed ' + str(s):>18}" for s in args.seeds) + f"{'mean':>18}")
for kind in ("fp32", "pipeline"):
	for r in ("full", "video"):
		cells = [scores[(kind, r, s)] for s in args.seeds]
		row = [f"{statistics.mean(c['psnr']):7.2f} / {statistics.mean(c['ssim']):.4f}" for c in cells]
		mean_psnr = statistics.mean(statistics.mean(c["psnr"]) for c in cells)
		mean_ssim = statistics.mean(statistics.mean(c["ssim"]) for c in cells)
		print(f"{kind:10} {r:6} " + "".join(f"{v:>18}" for v in row) + f"{mean_psnr:9.2f} / {mean_ssim:.4f}")

print("\npipeline PSNR, video minus full (same triplet, same seed):")
for seed in args.seeds:
	diff = [v - f for f, v in zip(scores[("pipeline", "full", seed)]["psnr"], scores[("pipeline", "video", seed)]["psnr"])]
	print(f"  seed {seed}: mean {statistics.mean(diff):+.3f} dB, sd {statistics.stdev(diff):.3f}, "
	      f"video better on {sum(d > 0 for d in diff)}/{len(diff)}")
if len(args.seeds) > 1:
	# Training noise: how much a model's mean PSNR moves with the seed alone, vs the video-minus-full difference.
	for r in ("full", "video"):
		means = [statistics.mean(scores[("pipeline", r, s)]["psnr"]) for s in args.seeds]
		print(f"{r} range, mean PSNR per seed: {' '.join(f'{m:.2f}' for m in means)}  (sd {statistics.stdev(means):.3f} dB)")
	per_seed = [statistics.mean(scores[("pipeline", "video", s)]["psnr"]) - statistics.mean(scores[("pipeline", "full", s)]["psnr"])
	            for s in args.seeds]
	print(f"video minus full over seeds: mean {statistics.mean(per_seed):+.3f} dB, sd {statistics.stdev(per_seed):.3f} dB "
	      f"(standard error {statistics.stdev(per_seed) / len(per_seed) ** 0.5:.3f} dB)")
