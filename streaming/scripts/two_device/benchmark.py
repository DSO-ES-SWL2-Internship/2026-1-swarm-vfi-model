# Runs receiver.py several times against the same sender and averages its performance numbers.
# The sender must be restarted between runs -- run it with LOOP=1 ./run_sender.sh on the Pi.

import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent.parent / "videos" / "benchmark"

PATTERNS = {
	"duration_s": r"Time from first to last frame written: ([\d.]+)s",
	"frames_arrived": r"Frames arrived: (\d+)",
	"buffers_written": r"buffers written: (\d+)",
	"infer_ms": r"Mean inference latency .*: ([\d.]+) ms/frame",
}

parser = argparse.ArgumentParser(description="Repeat receiver.py runs and average the results.")
parser.add_argument("technique")
parser.add_argument("--host", required=True, help="sender IP")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--connect-retries", type=int, default=15,
                    help="attempts per run while waiting for the looping sender to come back up")
args = parser.parse_args()

BENCH_DIR.mkdir(parents=True, exist_ok=True)


def run_once(run_index):
	output = BENCH_DIR / f"{args.technique}_run{run_index}.mp4"
	cmd = [sys.executable, str(SCRIPT_DIR / "receiver.py"), args.technique, "--host", args.host,
	       "--port", str(args.port), "--idle-timeout", "15", "--output", str(output)]
	for _ in range(args.connect_retries):
		result = subprocess.run(cmd, capture_output=True, text=True)
		if "failed to start" not in result.stdout:
			break
		time.sleep(2)  # sender is restarting between streams
	else:
		sys.exit(f"run {run_index}: could not connect to {args.host}:{args.port} -- is LOOP=1 ./run_sender.sh running?")

	metrics = {}
	for name, pattern in PATTERNS.items():
		match = re.search(pattern, result.stdout)
		metrics[name] = float(match.group(1)) if match else None
	if metrics["duration_s"] is None:
		print(result.stdout[-2000:])
		sys.exit(f"run {run_index}: receiver finished without writing any frames (output above)")
	metrics["output_fps"] = metrics["buffers_written"] / metrics["duration_s"]
	return metrics


results = []
for i in range(1, args.runs + 1):
	metrics = run_once(i)
	results.append(metrics)
	infer = f", inference {metrics['infer_ms']:.2f} ms" if metrics["infer_ms"] else ""
	print(f"run {i}/{args.runs}: {metrics['duration_s']:.2f}s, {metrics['frames_arrived']:.0f} arrived, "
	      f"{metrics['buffers_written']:.0f} written, {metrics['output_fps']:.1f} fps{infer}", flush=True)
	time.sleep(3)  # let the sender restart before the next run

csv_path = BENCH_DIR / f"results_{args.technique}.csv"
with open(csv_path, "w", newline="") as f:
	writer = csv.DictWriter(f, fieldnames=["run", *results[0].keys()])
	writer.writeheader()
	for i, metrics in enumerate(results, 1):
		writer.writerow({"run": i, **metrics})

print(f"\nSummary over {args.runs} runs (technique={args.technique}):")
for name in ["duration_s", "frames_arrived", "buffers_written", "output_fps", "infer_ms"]:
	values = [m[name] for m in results if m[name] is not None]
	if not values:
		continue
	stdev = statistics.stdev(values) if len(values) > 1 else 0.0
	print(f"  {name:16s} mean {statistics.mean(values):8.2f}   stdev {stdev:6.2f}   "
	      f"min {min(values):8.2f}   max {max(values):8.2f}")
print(f"Per-run results saved to {csv_path}")
