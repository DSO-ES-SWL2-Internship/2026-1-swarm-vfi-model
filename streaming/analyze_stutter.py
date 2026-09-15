import argparse
import csv
import statistics


def load_log(path):
	arrival, render = [], []
	with open(path, newline="") as f:
		for row in csv.DictReader(f):
			ts = float(row["timestamp"])
			if row["stage"] == "arrival":
				arrival.append(ts)
			elif row["stage"] == "render":
				render.append(ts)
	return arrival, render


def intervals(timestamps):
	# consecutive deltas -- timestamps are appended in arrival order, so this
	# is already in time order without needing an extra sort
	return [b - a for a, b in zip(timestamps, timestamps[1:])]


def percentile(sorted_values, p):
	if not sorted_values:
		return float("nan")
	k = (len(sorted_values) - 1) * (p / 100)
	f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
	if f == c:
		return sorted_values[f]
	return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def report(label, deltas, nominal):
	if len(deltas) < 2:
		print(f"\n--- {label}: not enough frames to compute stats ---")
		return

	mean = statistics.mean(deltas)
	stdev = statistics.stdev(deltas)
	cv = stdev / mean if mean else float("nan")
	sorted_deltas = sorted(deltas)
	p95 = percentile(sorted_deltas, 95)
	p99 = percentile(sorted_deltas, 99)

	outliers_1_5x = sum(1 for d in deltas if d > 1.5 * nominal)
	outliers_2x = sum(1 for d in deltas if d > 2.0 * nominal)
	# each interval that's N times the nominal represents roughly (N-1)
	# frames' worth of extra stall -- summed across all intervals gives a
	# single "how many frame-equivalents of stutter happened" number
	dropped_frame_equiv = sum(max(0, round(d / nominal) - 1) for d in deltas)

	print(f"\n--- {label} ({len(deltas) + 1} frames, {len(deltas)} intervals) ---")
	print(f"nominal interval:      {nominal * 1000:.2f} ms")
	print(f"mean interval:         {mean * 1000:.2f} ms")
	print(f"stddev:                {stdev * 1000:.2f} ms")
	print(f"coefficient of var.:   {cv:.3f}")
	print(f"95th percentile:       {p95 * 1000:.2f} ms")
	print(f"99th percentile:       {p99 * 1000:.2f} ms")
	print(f"outliers (>1.5x nom.): {outliers_1_5x} ({100 * outliers_1_5x / len(deltas):.1f}%)")
	print(f"outliers (>2x nom.):   {outliers_2x} ({100 * outliers_2x / len(deltas):.1f}%)")
	print(f"dropped-frame-equiv.:  {dropped_frame_equiv}")


parser = argparse.ArgumentParser(description="Compute stuttering metrics from a receiver.py frame_log CSV")
parser.add_argument("csv_path")
parser.add_argument("--fps", type=float, default=None,
                     help="expected source framerate, used to derive the nominal inter-frame interval for the "
                          "'arrival' stage. If omitted, the nominal interval is the MEDIAN observed interval for "
                          "that stage instead (robust to outliers, but only meaningful once you already have a "
                          "run to look at -- pass --fps explicitly when you know the true target rate).")
args = parser.parse_args()

arrival, render = load_log(args.csv_path)
arrival_deltas = intervals(arrival)
render_deltas = intervals(render)

arrival_nominal = (1 / args.fps) if args.fps else statistics.median(arrival_deltas)
# render stage is post-VFI, which for the techniques in receiver.py doubles the
# framerate (one interpolated + one real frame per arrival) -- halve the arrival
# nominal accordingly, unless an explicit render-side rate should be treated
# differently for a technique that doesn't double the rate (e.g. passthrough)
render_nominal = (arrival_nominal / 2) if args.fps else statistics.median(render_deltas)

report("arrival (pre-VFI, network+decode timing)", arrival_deltas, arrival_nominal)
report("render (post-VFI, what's actually displayed)", render_deltas, render_nominal)
