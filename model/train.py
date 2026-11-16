"""Trains the VFI U-Net. The defaults are the recipe of the deployed model (A3):

  residual output (average of the two frames + a learned correction), width 1.0, Charbonnier loss;
  data: Blender-film windows (more motion) mixed with Vimeo-90K triplets, 20% Vimeo; flips and time reversal;
  3000 steps of 8 (12 epochs of 250), Adam 1e-3 with a cosine decay, seed 0 (about 20 min on this VM's CPU).

The other flags are the ablations behind that choice, one change at a time:
  --plain             a plain sigmoid output instead of blend + correction
  --vimeo-fraction 1  Vimeo only (no Blender windows)
  --width 1.5         wider network (no gain; --width 2 collapsed to the plain blend)
  --grad-weight 0.5   gradient (sharpness) loss (no gain)
(run_experiments.sh lists the runs.) A run saves the model, prints its scores on the held-out sets, and that is all:
python3 evaluate.py <model> adds LPIPS; python3 export_tflite.py makes the INT8 file for the NPU.

Usage (on the VM): python3 train.py                      -> artifacts/vfi_unet.keras
                   python3 train.py --name a5 --width 1.5   -> artifacts/experiments/a5.keras (+ a line in results.txt)
"""
import argparse
import time
from pathlib import Path

from tensorflow import keras

import data
import evaluate
import model as vfi


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--out", type=Path, help="where to save the model (default artifacts/vfi_unet.keras)")
	ap.add_argument("--name", help="an ablation run: save as artifacts/experiments/NAME.keras and log to results.txt")
	ap.add_argument("--plain", action="store_true")
	ap.add_argument("--width", type=float, default=1.0)
	ap.add_argument("--grad-weight", type=float, default=0.0)
	ap.add_argument("--vimeo-fraction", type=float, default=0.2)
	ap.add_argument("--epochs", type=int, default=12)
	ap.add_argument("--steps", type=int, default=250, help="batches of 8 per epoch")
	ap.add_argument("--seed", type=int, default=0)
	args = ap.parse_args()
	out = args.out or (vfi.EXPERIMENTS_DIR / f"{args.name}.keras" if args.name else vfi.MODEL_PATH)
	out.parent.mkdir(parents=True, exist_ok=True)

	keras.utils.set_random_seed(args.seed)
	model = vfi.build_model(args.width, residual=not args.plain)
	total = args.epochs * args.steps
	model.compile(keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, total, alpha=0.05)),
	              loss=vfi.make_loss(args.grad_weight))
	start = time.time()
	model.fit(data.train_dataset(args.vimeo_fraction, args.seed), steps_per_epoch=args.steps, epochs=args.epochs, verbose=2)
	model.save(out)
	line = evaluate.summary(out.stem, evaluate.psnr_ssim(lambda x: model.predict(x, verbose=0), data.test_sets()))
	line += f"  [{model.count_params()} params, {(time.time() - start) / 60:.0f} min; {vars(args)}]"
	print(f"saved {out}\n{line}")
	if args.name:
		with open(vfi.EXPERIMENTS_DIR / "results.txt", "a") as f:
			f.write(line + "\n")


if __name__ == "__main__":
	main()
