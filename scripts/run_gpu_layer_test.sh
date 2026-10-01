#!/usr/bin/env bash

# Benchmarks every model from gpu_layer_test.py (plus the deployed toy_unet_int8.tflite) on the CPU,
# NPU and GPU with benchmark_model, and writes a summary table of average inference times.
#
# Run on the iMX8, after copying artifacts/gpu_layer_test/ over. Takes a few minutes: GPU runs of the
# full U-Net are ~9 s each, so the GPU gets fewer runs.
#
# Usage: ./run_gpu_layer_test.sh      -> gpu_layer_test_results.txt (+ full log)
set -uo pipefail
cd "$(dirname "$0")"

BM=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model
DELEGATE=/usr/lib/libvx_delegate.so
MODELS_DIR=../artifacts/gpu_layer_test
RESULTS=gpu_layer_test_results.txt
LOG=gpu_layer_test_log.txt
: > "$LOG"

# Extract average time taken from the logs
avg_ms() {  # $1 = log text -> "Inference (avg)" in ms, or FAILED
	local us
	us=$(grep -o "Inference (avg): [0-9.e+]*" <<< "$1" | awk '{print $3}')
	[ -n "$us" ] && awk -v u="$us" 'BEGIN { printf "%.2f", u / 1000 }' || echo "FAILED"
}

printf "%-34s %10s %10s %10s %10s\n" "model" "CPU ms" "NPU ms" "GPU ms" "GPU/CPU" | tee "$RESULTS"
# Do for all models and also toy_unet_int8 model for sanity check
for model in "$MODELS_DIR"/*.tflite ../artifacts/toy_unet_int8.tflite; do 
	name=$(basename "$model" .tflite)

    # Create aliases for CPU, NPU and GPU runs
	cpu_out=$($BM --graph="$model" --num_runs=10 2>&1)
	npu_out=$($BM --graph="$model" --num_runs=10 --external_delegate_path=$DELEGATE 2>&1)
	gpu_out=$(USE_GPU_INFERENCE=1 $BM --graph="$model" --num_runs=3 --external_delegate_path=$DELEGATE 2>&1)

    # Run and print out log, sent to log file
	printf "==== %s\n--- CPU\n%s\n--- NPU\n%s\n--- GPU\n%s\n" "$name" "$cpu_out" "$npu_out" "$gpu_out" >> "$LOG"
    
    # Print row of results
	cpu=$(avg_ms "$cpu_out"); npu=$(avg_ms "$npu_out"); gpu=$(avg_ms "$gpu_out")
	ratio=$(awk -v c="$cpu" -v g="$gpu" 'BEGIN { if (c + 0 > 0 && g + 0 > 0) printf "%.1fx", g / c; else print "-" }')
	printf "%-34s %10s %10s %10s %10s\n" "$name" "$cpu" "$npu" "$gpu" "$ratio" | tee -a "$RESULTS"
done

echo "Summary: $RESULTS   Full benchmark_model output: $LOG"
