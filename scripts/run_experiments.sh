#!/usr/bin/env bash
# The model-quality experiments of 12-13 Nov 2026, one change at a time (train_better.py; ~20-30 min each on this VM's CPU).
# Scores (PSNR, SSIM, LPIPS) are then read with score_sintel.py on held-out Sintel windows:
#   python3 score_sintel.py a0_base_vimeo a1_base_newdata a2_res_vimeo a3_res_newdata ...
#
#   2x2: architecture (current U-Net / residual output) by data (Vimeo only / Blender motion data + Vimeo)
#     a0  current U-Net, Vimeo only (the control)      a1  current U-Net + Blender data
#     a2  residual output, Vimeo only                  a3  residual output + Blender data   <- the candidate
#   seed check (a gain only counts if it beats the spread between seeds): a3 and a0 again with --seed 1
#   from a3, one change each: a4 width 1.5x, a5 gradient (sharpness) loss 0.5, a6 width 2x
# Usage: ./run_experiments.sh [2x2 | seed | extras]     (several runs back to back; log in ../artifacts/better/ladder.log)
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2
PY="${PY:-$HOME/vfi-env/bin/python}"  # the venv with TensorFlow, also when started without it activated
run() { "$PY" train_better.py "$@" >> ../artifacts/better/ladder.log 2>&1; }

case "${1:-2x2}" in
	2x2)
		run --name a0_base_vimeo --vimeo-fraction 1.0
		run --name a1_base_newdata
		run --name a2_res_vimeo --residual --vimeo-fraction 1.0
		run --name a3_res_newdata --residual ;;
	seed)
		run --name a3_res_newdata_s1 --residual --seed 1
		run --name a0_base_vimeo_s1 --vimeo-fraction 1.0 --seed 1 ;;
	extras)
		run --name a4_res_w15 --residual --width 1.5
		run --name a5_res_grad --residual --grad-weight 0.5
		run --name a6_res_w2 --residual --width 2 ;;
	*) sed -n '2,13p' "$0"; exit 1 ;;
esac
