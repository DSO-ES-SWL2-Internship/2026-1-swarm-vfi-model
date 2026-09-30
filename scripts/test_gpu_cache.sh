BM=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model
MODEL=~/projects/intern1-swarm/machine-learning/artifacts/toy_unet_int8.tflite
export VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1
export VIV_VX_CACHE_BINARY_GRAPH_DIR=/root/vx_cache && mkdir -p /root/vx_cache

# 1. Same GPU run with graph caching on. Run it TWICE: the second run should load the cached graph.
USE_GPU_INFERENCE=1 $BM --graph=$MODEL --num_runs=5 --external_delegate_path=/usr/lib/libvx_delegate.so
USE_GPU_INFERENCE=1 $BM --graph=$MODEL --num_runs=5 --external_delegate_path=/usr/lib/libvx_delegate.so
ls -lh /root/vx_cache      # a *.nb file here = the compiled graph was cached

# 2. Where does the GPU's time go? Per-operation breakdown
USE_GPU_INFERENCE=1 $BM --graph=$MODEL --num_runs=3 --external_delegate_path=/usr/lib/libvx_delegate.so \
  --enable_op_profiling=true 2>&1 | tail -40
