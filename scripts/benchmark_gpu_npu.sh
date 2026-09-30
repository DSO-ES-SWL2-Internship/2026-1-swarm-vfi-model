BM=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model
MODEL=~/projects/intern1-swarm/machine-learning/artifacts/toy_unet_int8.tflite

# 1. CPU only
$BM --graph=$MODEL --num_runs=50

# 2. NPU (default for VX delegate on this chip)
$BM --graph=$MODEL --num_runs=50 --external_delegate_path=/usr/lib/libvx_delegate.so

# 3. GPU
USE_GPU_INFERENCE=1 $BM --graph=$MODEL --num_runs=50 --external_delegate_path=/usr/lib/libvx_delegate.so
