# experiments/ (streaming)

Benchmarks and checks behind the video-pipeline results: what offloading each step to the VPU, GPU and NPU buys. They
are one-off measurements, kept so a number can be re-run; the demo does not use them. Run them on the iMX8 unless noted.
Results go to `../videos/benchmark/` and `../../outputs/`.

| Script | What it measured |
| --- | --- |
| `relay_bench.py` | The real relay with the model, one offload at a time (all CPU, + VPU decode, + GPU scale, + NPU, + VPU encode), each without and with queues -> `outputs/relay_bench/` (slides: the pipeline offloading chart, 176 -> 17.6 ms per frame) |
| `pipeline_bench.py` | decode -> scale -> encode on CPU / VPU / G2D, without and with queues, with per-step predictions (why queues help) |
| `codec_suite.py` | Every decoder / scaler / encoder combination at 480p, 720p and 1080p: speed, CPU time, utilisation; `plot_codec_suite.py` and `plot_utilisation.py` plot it on the VM (needs matplotlib) |
| `trace_buffers.py`, `scaler_tlb.py` | Frame memory per hand-over between units; why the CPU scaler is slower on VPU frames |
| `test_g2d_colour.sh` | Which colorimetry setting gives the right grey on the G2D path (the G2D is no longer in the relay) |
| `compare_videos.sh` | Plays an original and a received video side by side |

The G2D scaler appears in these benchmarks only as a measured alternative: the relay uses the GPU (OpenCL) scaler.
