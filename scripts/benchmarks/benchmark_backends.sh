#!/usr/bin/env bash

# Benchmarks TFLite models on the iMX8's CPU, NPU and GPU with benchmark_model and writes a summary
# table of average inference times. Replaces benchmark_gpu_npu.sh, benchmark_mobilenet.sh and
# run_gpu_layer_test.sh, which ran the same three commands on different models.
#
# Run on the iMX8. benchmark_model feeds random inputs; that is fine for timing because a CNN does the
# same work whatever the pixel values (live pipeline 23.10 ms == benchmark 23.1 ms, per-channel model).
#
# Usage: 
#   ./benchmark_backends.sh [model.tflite ...]     (default: the deployed toy U-Net)
#   ./benchmark_backends.sh /usr/bin/tensorflow-lite-2.19.0/examples/mobilenet_v1_1.0_224_quant.tflite (MobileNet CNN evaluation)
#   ./benchmark_backends.sh ../../artifacts/gpu_layer_test/*.tflite ../../artifacts/toy_unet_int8.tflite (Single layer by type evaluation)
#                                     (copy artifacts/gpu_layer_test/ from the VM first; takes a few minutes)
# Output: outputs/benchmarks/backends_<timestamp>.txt (table) and .log (full benchmark_model output).
# RUNS / GPU_RUNS override the run counts; the GPU gets fewer because the U-Net takes ~9 s per run there.
set -uo pipefail
cd "$(dirname "$0")"

BM=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model
DELEGATE=/usr/lib/libvx_delegate.so
RUNS=${RUNS:-20}
GPU_RUNS=${GPU_RUNS:-3}
OUT_DIR=../../outputs/benchmarks
STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS=$OUT_DIR/backends_$STAMP.txt
LOG=$OUT_DIR/backends_$STAMP.log
mkdir -p "$OUT_DIR"
: > "$LOG"

[ $# -eq 0 ] && set -- ../../artifacts/toy_unet_int8.tflite

# Extract average time taken from the logs
avg_ms() {  # $1 = log text -> "Inference (avg)" in ms, or FAILED
	local us
	us=$(grep -o "Inference (avg): [0-9.e+]*" <<< "$1" | awk '{print $3}')
	[ -n "$us" ] && awk -v u="$us" 'BEGIN { printf "%.2f", u / 1000 }' || echo "FAILED"
}

printf "%-34s %10s %10s %10s %10s\n" "model" "CPU ms" "NPU ms" "GPU ms" "GPU/CPU" | tee "$RESULTS"
for model in "$@"; do
	name=$(basename "$model" .tflite)

	# CPU (no delegate), NPU (VX delegate's default on this chip), GPU (same delegate, redirected)
	cpu_out=$($BM --graph="$model" --num_runs="$RUNS" 2>&1)
	npu_out=$($BM --graph="$model" --num_runs="$RUNS" --external_delegate_path=$DELEGATE 2>&1)
	gpu_out=$(USE_GPU_INFERENCE=1 $BM --graph="$model" --num_runs="$GPU_RUNS" --external_delegate_path=$DELEGATE 2>&1)

	# Keep the full output for each backend in the log file
	printf "==== %s\n--- CPU\n%s\n--- NPU\n%s\n--- GPU\n%s\n" "$name" "$cpu_out" "$npu_out" "$gpu_out" >> "$LOG"

	# Print row of results
	cpu=$(avg_ms "$cpu_out"); npu=$(avg_ms "$npu_out"); gpu=$(avg_ms "$gpu_out")
	ratio=$(awk -v c="$cpu" -v g="$gpu" 'BEGIN { if (c + 0 > 0 && g + 0 > 0) printf "%.1fx", g / c; else print "-" }')
	printf "%-34s %10s %10s %10s %10s\n" "$name" "$cpu" "$npu" "$gpu" "$ratio" | tee -a "$RESULTS"
done

echo "Summary: $RESULTS   Full benchmark_model output: $LOG"
