BM=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model
M=/usr/bin/tensorflow-lite-2.19.0/examples/mobilenet_v1_1.0_224_quant.tflite

# Run test on CPU
$BM --graph=$M --num_runs=50

# Run test on NPU
$BM --graph=$M --num_runs=50 --external_delegate_path=/usr/lib/libvx_delegate.so

# Run test on GPU
USE_GPU_INFERENCE=1 $BM --graph=$M --num_runs==20 --external_delegate_path=/usr/lib/libvx_delegate.so
