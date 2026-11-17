#!/usr/bin/env bash
# The model-quality experiments of 12-13 Nov 2026, one change at a time (train.py; ~20-30 min each on this VM's CPU).
# Models go to artifacts/experiments/, one line per run in artifacts/experiments/results.txt. Perceptual scores (LPIPS):
#   python3 evaluate.py ../artifacts/experiments/a0_base_vimeo.keras ../artifacts/experiments/a3_res_newdata.keras ...
#
#   2x2: output (plain sigmoid / residual: blend + correction) by data (Vimeo only / Blender motion data + Vimeo)
#     a0  plain, Vimeo only (the control)      a1  plain + Blender data
#     a2  residual, Vimeo only                 a3  residual + Blender data   <- the deployed model (train.py's defaults)
#   seed check (a gain only counts if it beats the spread between seeds): a3 and a0 again with --seed 1
#   from a3, one change each: a4 width 1.5x, a5 gradient (sharpness) loss 0.5, a6 width 2x (collapsed to the blend)
# Usage: experiments/run_experiments.sh [2x2 | seed | extras]     (several runs back to back; log in ../artifacts/experiments/ladder.log)
cd "$(dirname "$0")/.."  # model/, where train.py is
export TF_CPP_MIN_LOG_LEVEL=2
PY="${PY:-$HOME/vfi-env/bin/python}"  # the venv with TensorFlow, also when started without it activated
mkdir -p ../artifacts/experiments
run() { "$PY" train.py "$@" >> ../artifacts/experiments/ladder.log 2>&1; }

case "${1:-2x2}" in
	2x2)
		run --name a0_base_vimeo --plain --vimeo-fraction 1.0
		run --name a1_base_newdata --plain
		run --name a2_res_vimeo --vimeo-fraction 1.0
		run --name a3_res_newdata ;;
	seed)
		run --name a3_res_newdata_s1 --seed 1
		run --name a0_base_vimeo_s1 --plain --vimeo-fraction 1.0 --seed 1 ;;
	extras)
		run --name a4_res_w15 --width 1.5
		run --name a5_res_grad --grad-weight 0.5
		run --name a6_res_w2 --width 2 ;;
	*) sed -n '2,12p' "$0"; exit 1 ;;
esac
