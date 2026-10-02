# Plots codec_suite.py results (run on the VM -- needs matplotlib). 2x2 grid:
#   rows    = speed (fps) and CPU time
#   columns = encoder (CPU MPEG-4 vs VPU H.264)
#   lines   = the four decode -> scale combinations, against source resolution, mean +/- stdev
# Usage: python3 plot_codec_suite.py [results.csv]   (defaults to the newest codec_suite_*.csv run summary)

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "videos" / "benchmark"
RESOLUTIONS = ["480p", "720p", "1080p"]
REAL_TIME_FPS = [(24, "source: 24 fps"), (60, "stress test: 60 fps")]  # reference lines on the speed panels

# Fixed colour per decode -> scale combination (same meaning in every panel and in earlier charts).
SERIES = [
	("cpu", "cpu", "CPU decode → CPU scale", "#2a78d6"),
	("vpu", "cpu", "VPU decode → CPU scale", "#eb6834"),
	("vpu", "g2d", "VPU decode → G2D scale", "#1baf7a"),
	("cpu", "g2d", "CPU decode → G2D scale", "#eda100"),
]
ENCODERS = [("cpu", "CPU encode (MPEG-4)"), ("vpu", "VPU encode (H.264)")]
SURFACE, TEXT, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"

if len(sys.argv) > 1:
	csv_path = Path(sys.argv[1])
else:
	# Newest run summary; the per-sample *_timeseries.csv files are a different format.
	found = sorted((p for p in RESULTS_DIR.glob("codec_suite_*.csv") if not p.stem.endswith("_timeseries")),
	               key=lambda p: p.stat().st_mtime)
	if not found:
		raise SystemExit(f"no codec_suite_*.csv in {RESULTS_DIR}")
	csv_path = found[-1]

runs = defaultdict(list)  # (resolution, decoder, scaler, encoder) -> list of (fps, cpu_s)
repeats, modes = set(), set()
with open(csv_path) as f:
	for r in csv.DictReader(f):
		repeats.add(r["repeat"])
		modes.add(r.get("mode", "flatout"))  # files from before --pace have no mode column
		if r["status"] == "ok":
			runs[(r["resolution"], r["decoder"], r["scaler"], r["encoder"])].append((float(r["fps"]), float(r["cpu_s"])))
resolutions = [res for res in RESOLUTIONS if any(k[0] == res for k in runs)]
x = list(range(len(resolutions)))


def stats(dec, scale, enc, metric):
	means, sds = [], []
	for res in resolutions:
		vals = [v[metric] for v in runs.get((res, dec, scale, enc), [])]
		means.append(statistics.mean(vals) if vals else float("nan"))
		sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
	return means, sds


def spread_labels(values, min_gap):
	# Nudges end-of-line label positions apart so they don't overlap (keeps their order).
	order = sorted(range(len(values)), key=lambda i: values[i])
	placed = list(values)
	for a, b in zip(order, order[1:]):
		if placed[b] - placed[a] < min_gap:
			placed[b] = placed[a] + min_gap
	return placed


plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor=SURFACE, sharey="row")

for row, (metric, ylabel, unit) in enumerate([(0, "frames per second", "fps"), (1, "CPU seconds (whole video)", "s")]):
	ymax = max((max(v[metric] for v in vals) for vals in runs.values()), default=1) * 1.12
	for col, (enc, enc_label) in enumerate(ENCODERS):
		ax = axes[row][col]
		ends, colors = [], []
		for dec, scale, label, color in SERIES:
			means, sds = stats(dec, scale, enc, metric)
			ax.errorbar(x, means, yerr=sds, color=color, linewidth=2.5, marker="o", markersize=8,
			            markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=4, elinewidth=1.5, zorder=3)
			ends.append(means[-1]); colors.append(color)
		# Labels may be nudged apart; a thin leader line in the series colour ties each one back to
		# its line (no marker at the label end -- it would read as an extra data point).
		for y, value, color in zip(spread_labels(ends, ymax * 0.07), ends, colors):
			if value != value:  # NaN: configuration failed at this resolution
				continue
			ax.plot([x[-1] + 0.05, x[-1] + 0.14, x[-1] + 0.2], [value, y, y], color=color, linewidth=1.2,
			        zorder=2, clip_on=False)
			ax.text(x[-1] + 0.23, y, f"{value:.0f} {unit}", va="center", fontsize=11, color=TEXT)
		if metric == 0:
			for fps, label in REAL_TIME_FPS:
				ax.axhline(fps, color=MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
				ax.text(-0.15, fps - ymax * 0.045, label, fontsize=10, color=MUTED)  # below the line: lines start near 60
		ax.set_title(f"{'Speed' if metric == 0 else 'CPU time'} · {enc_label}", loc="left",
		             fontsize=13, fontweight="bold", color=TEXT)
		ax.set_xticks(x, resolutions)
		ax.set_xlim(-0.25, x[-1] + 0.7)
		ax.set_ylim(0, ymax)
		if col == 0:
			ax.set_ylabel(ylabel, color=MUTED)
		ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
		for side in ["top", "right", "left"]:
			ax.spines[side].set_visible(False)
		ax.spines["bottom"].set_color(GRID)
		ax.tick_params(colors=MUTED, length=0)

fig.suptitle("Decode → scale to 448×256 grey → encode on the i.MX8M Plus, by source resolution",
             x=0.01, y=0.985, ha="left", fontsize=16, fontweight="bold", color=TEXT)
mode = modes.pop() if len(modes) == 1 else "mixed"
how = f"played in real time at {mode[5:]} fps" if mode.startswith("paced") else "run flat out"
fig.text(0.01, 0.945, f"Sintel trailer, 1251 frames, {how}; mean ± stdev of {len(repeats)} runs. "
         "CPU time = user + system seconds across all cores. CPU encoder is MPEG-4 (cheaper than H.264).",
         ha="left", fontsize=10.5, color=MUTED)
handles = [plt.Line2D([], [], color=c, linewidth=2.5, marker="o", markersize=8, markeredgecolor=SURFACE, label=l)
           for _, _, l, c in SERIES]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11.5, labelcolor=TEXT)
fig.subplots_adjust(left=0.07, right=0.94, top=0.88, bottom=0.1, hspace=0.32, wspace=0.18)

out_path = csv_path.with_suffix(".png")
fig.savefig(out_path, dpi=200, facecolor=SURFACE)
print(f"Saved {out_path}")
