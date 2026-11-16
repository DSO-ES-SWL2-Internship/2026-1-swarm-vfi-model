# Checks the Python code on the "NPU Usage for ML Model" slide, and that the C++ version gives the same answer.
# Runs the slide's Python on the input that npu_infer_test saved, then compares its output with the C++ output.
#
# Run on the iMX8, after npu_infer_test:
#   python3 compare_with_python.py MODEL.tflite [--cpu] [--out DIR]
# Same --cpu and --out as npu_infer_test. On the same unit the outputs should match exactly.

import argparse
import sys
import time

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

VX_DELEGATE = "/usr/lib/libvx_delegate.so"
CPU_THREADS = 4

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--cpu", action="store_true", help="no delegate, 4 CPU threads")
parser.add_argument("--out", default="/tmp/npu_infer_test", help="where npu_infer_test saved its files")
parser.add_argument("--runs", type=int, default=50)
args = parser.parse_args()
target = "cpu" if args.cpu else "npu"

# Slide row 1 -- set up: the interpreter with the model and the NPU delegate
delegates = [] if args.cpu else [load_delegate(VX_DELEGATE)]
interp = Interpreter(model_path=args.model, experimental_delegates=delegates,
                     num_threads=CPU_THREADS if args.cpu else None)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"model {args.model} on the {target}")
print(f"input  {inp['dtype'].__name__} {inp['shape'].tolist()}, scale {inp['quantization'][0]:.6g}, "
      f"zero point {inp['quantization'][1]}")
print(f"output {out['dtype'].__name__} {out['shape'].tolist()}, scale {out['quantization'][0]:.6g}, "
      f"zero point {out['quantization'][1]}")

# Slide row 2 -- warm up: one inference so the NPU graph is compiled and cached
start = time.perf_counter()
dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
interp.set_tensor(inp["index"], dummy)
interp.invoke()
print(f"warm-up: {(time.perf_counter() - start) * 1000:.1f} ms")

# Slide row 3 -- run: load the inputs -> run inference -> read the outputs, on the C++ run's input
frames = np.fromfile(f"{args.out}/input.bin", dtype=inp["dtype"]).reshape(inp["shape"])
total = 0.0
for _ in range(args.runs):
	start = time.perf_counter()
	interp.set_tensor(inp["index"], frames)
	interp.invoke()
	total += time.perf_counter() - start
result = interp.get_tensor(out["index"])
print(f"inference: {total / args.runs * 1000:.2f} ms mean over {args.runs} runs (Python)")

cpp = np.fromfile(f"{args.out}/output_cpp_{target}.bin", dtype=out["dtype"]).reshape(out["shape"])
diff = np.abs(result.astype(np.int16) - cpp.astype(np.int16))
print(f"C++ vs Python output: {np.count_nonzero(diff)} of {diff.size} values differ, largest difference {diff.max()}")
if diff.max() == 0:
	print("MATCH: the C++ and Python code give the same output")
else:
	sys.exit("MISMATCH: see the numbers above")
