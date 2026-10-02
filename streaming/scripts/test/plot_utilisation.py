# Plots the resource-utilisation side of a codec_suite.py run (run on the VM -- needs matplotlib):
#   <run>_utilisation.png  3x2 grid, columns = encoder, lines = decode -> scale combination:
#                          lag behind real time (paced runs only), CPU % of all 4 cores, and busiest-core %
#   <run>_cores.png        per-core heat strips over time for four configs at one resolution: shows the
#                          work sitting on one core at a time (a single busy thread), which per-core
#                          means hide because Linux moves the thread between cores
# "Busiest core" = per 0.1 s sample, the most loaded core's busy %; averaged over the run (first 1 s
# excluded, as in codec_suite.py). Near 100% = one thread is saturated, whatever the all-core average says.
# Usage: python3 plot_utilisation.py [results.csv] [--resolution 1080p]   (defaults to the newest run)

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "videos" / "benchmark"
RESOLUTIONS = ["480p", "720p", "1080p"]
STARTUP_S = 1.0  # same exclusion as codec_suite.py's summary

# Same colour per decode -> scale combination as plot_codec_suite.py.
SERIES = [
	("cpu", "cpu", "CPU decode → CPU scale", "#2a78d6"),
	("vpu", "cpu", "VPU decode → CPU scale", "#eb6834"),
	("vpu", "g2d", "VPU decode → G2D scale", "#1baf7a"),
	("cpu", "g2d", "CPU decode → G2D scale", "#eda100"),
]
ENCODERS = [("cpu", "CPU encode (MPEG-4)"), ("vpu", "VPU encode (H.264)")]
# Heat-strip configs: best to worst hardware use, all with the VPU encoder except the all-CPU baseline.
STRIP_CONFIGS = ["vpu -> g2d -> vpu", "cpu -> g2d -> vpu", "vpu -> cpu -> vpu", "cpu -> cpu -> cpu"]
SURFACE, TEXT, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", type=Path)
parser.add_argument("--resolution", default="1080p", help="resolution for the per-core heat strips")
args = parser.parse_args()
if args.csv:
	csv_path = args.csv
else:
	found = sorted((p for p in RESULTS_DIR.glob("codec_suite_*.csv") if not p.stem.endswith("_timeseries")),
	               key=lambda p: p.stat().st_mtime)
	if not found:
		raise SystemExit(f"no codec_suite_*.csv in {RESULTS_DIR}")
	csv_path = found[-1]
series_path = csv_path.with_name(csv_path.stem + "_timeseries.csv")
if not series_path.exists():
	raise SystemExit(f"{series_path.name} not found -- utilisation needs a run from codec_suite.py v2 or later")

with open(csv_path) as f:
	rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
mode = rows[0]["mode"]
paced = mode.startswith("paced")

# Busiest core per sample, from the time series.
samples = defaultdict(list)  # (resolution, config, repeat) -> list of (t, [core %...])
with open(series_path) as f:
	for s in csv.DictReader(f):
		cores = [float(v) for k, v in s.items() if k.startswith("cpu")]
		samples[(s["resolution"], s["config"], s["repeat"])].append((float(s["t"]), cores))
busiest = {key: statistics.mean(max(c) for t, c in ss if t >= STARTUP_S) for key, ss in samples.items()}

runs = defaultdict(list)  # (resolution, config) -> list of metric dicts
for r in rows:
	runs[(r["resolution"], r["config"])].append({
		"lag": float(r["lag_s"]) if r.get("lag_s") else float("nan"),
		"cpu_all": float(r["cpu_util_mean_pct"]),
		"cpu_top": busiest.get((r["resolution"], r["config"], r["repeat"]), float("nan")),
	})
resolutions = [res for res in RESOLUTIONS if any(k[0] == res for k in runs)]
x = list(range(len(resolutions)))
repeats = max(len(v) for v in runs.values())

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})


def style(ax, title, ylabel, ymax, first_col):
	ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT)
	ax.set_xticks(x, resolutions)
	ax.set_xlim(-0.25, x[-1] + 0.7)
	ax.set_ylim(0, ymax)
	if first_col:
		ax.set_ylabel(ylabel, color=MUTED)
	ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
	for side in ["top", "right", "left"]:
		ax.spines[side].set_visible(False)
	ax.spines["bottom"].set_color(GRID)
	ax.tick_params(colors=MUTED, length=0)


def spread(values, gap):
	# Nudges end-of-line label positions apart so they don't overlap (keeps their order).
	order = sorted(range(len(values)), key=lambda i: values[i])
	placed = list(values)
	for a, b in zip(order, order[1:]):
		if placed[b] - placed[a] < gap:
			placed[b] = placed[a] + gap
	return placed


# --- Figure 1: lag, all-core CPU %, busiest-core % ---------------------------------------------
metrics = ([("lag", "Lag behind real time", "seconds late (20.9 s video)", "s")] if paced else []) + [
	("cpu_all", "CPU, all 4 cores", "% of all 4 cores busy", "%"),
	("cpu_top", "Busiest core", "% busy (most loaded core)", "%"),
]
fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 4.4 * len(metrics) + 1.2), facecolor=SURFACE,
                         sharey="row", squeeze=False)
for row, (metric, title, ylabel, unit) in enumerate(metrics):
	ymax = 108 if unit == "%" else max(m[metric] for v in runs.values() for m in v) * 1.15
	for col, (enc, enc_label) in enumerate(ENCODERS):
		ax = axes[row][col]
		ends, colors = [], []
		for dec, scale, _, color in SERIES:
			config = f"{dec} -> {scale} -> {enc}"
			means = [statistics.mean(m[metric] for m in runs[(res, config)]) if runs.get((res, config)) else float("nan")
			         for res in resolutions]
			ax.plot(x, means, color=color, linewidth=2.5, marker="o", markersize=8, markeredgecolor=SURFACE,
			        markeredgewidth=1.5, zorder=3)
			ends.append(means[-1]); colors.append(color)
		for y, value, color in zip(spread(ends, ymax * 0.07), ends, colors):
			ax.plot([x[-1] + 0.05, x[-1] + 0.14, x[-1] + 0.2], [value, y, y], color=color, linewidth=1.2,
			        zorder=2, clip_on=False)
			ax.text(x[-1] + 0.23, y, f"{value:.0f}{'' if unit == '%' else ' '}{unit}", va="center", fontsize=11, color=TEXT)
		if metric == "cpu_all":
			ax.axhline(25, color=MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
			ax.text(x[-1] + 0.23, 25 + 108 * 0.07, "25% = one\ncore's worth", va="bottom", fontsize=10, color=MUTED)
		style(ax, f"{title} · {enc_label}", ylabel, ymax, col == 0)

how = f"played in real time at {mode[5:]} fps" if paced else "run flat out"
fig.suptitle("Where the CPU time goes: whole board vs busiest core", x=0.01, y=0.99, ha="left",
             fontsize=16, fontweight="bold", color=TEXT)
fig.text(0.01, 1 - 0.95 / fig.get_figheight(), 
         f"Sintel trailer, 1251 frames, {how}; mean of {repeats} runs. CPU sampled every 0.1 s from /proc/stat, first 1 s excluded.\n"
         "Busiest core = the most loaded core in each sample. Low all-core % with one core near 100% = a single thread is the bottleneck.",
         ha="left", fontsize=10.5, color=MUTED)
handles = [plt.Line2D([], [], color=c, linewidth=2.5, marker="o", markersize=8, markeredgecolor=SURFACE, label=l)
           for _, _, l, c in SERIES]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11.5, labelcolor=TEXT)
fig.subplots_adjust(left=0.07, right=0.94, top=1 - 1.45 / fig.get_figheight(), bottom=0.6 / fig.get_figheight() + 0.03,
                    hspace=0.38, wspace=0.18)
out1 = csv_path.with_name(csv_path.stem + "_utilisation.png")
fig.savefig(out1, dpi=200, facecolor=SURFACE)
print(f"Saved {out1}")

# --- Figure 2: per-core heat strips over time -------------------------------------------------
lag_of = {k: statistics.mean(m["lag"] for m in v) for k, v in runs.items()}
fig, axes = plt.subplots(len(STRIP_CONFIGS), 1, figsize=(14, 1.7 * len(STRIP_CONFIGS) + 1.6), facecolor=SURFACE,
                         sharex=True)
image = None
for ax, config in zip(axes, STRIP_CONFIGS):
	ss = samples.get((args.resolution, config, "1"), [])
	if not ss:
		ax.set_axis_off()
		continue
	t = [s[0] for s in ss]
	grid = [[s[1][core] for s in ss] for core in range(len(ss[0][1]))]
	dt = t[1] - t[0] if len(t) > 1 else 0.1
	image = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=100, interpolation="nearest",
	                  extent=(t[0] - dt / 2, t[-1] + dt / 2, len(grid) - 0.5, -0.5))
	ax.set_yticks(range(len(grid)), [f"core {c}" for c in range(len(grid))], fontsize=9.5, color=MUTED)
	ax.tick_params(length=0, colors=MUTED)
	for side in ax.spines.values():
		side.set_visible(False)
	key = (args.resolution, config)
	extra = f"  ·  lag {lag_of[key]:+.1f} s" if paced and key in lag_of else ""
	top = busiest.get((args.resolution, config, "1"), float("nan"))
	ax.set_title(f"{config.replace('->', '→')}   (busiest core {top:.0f}%{extra})", loc="left",
	             fontsize=12, fontweight="bold", color=TEXT)
axes[-1].set_xlabel("seconds into the run", color=MUTED)
fig.suptitle(f"Per-core load over time, {args.resolution}: one busy core at a time = a single-threaded pipeline",
             x=0.01, y=0.99, ha="left", fontsize=15, fontweight="bold", color=TEXT)
fig.text(0.01, 1 - 0.7 / fig.get_figheight(), "Darker = busier (0-100% per core, 0.1 s samples, first repeat). "
         "The dark band moves between cores because Linux migrates the thread; per-core averages smear it out.",
         ha="left", fontsize=10.5, color=MUTED)
fig.subplots_adjust(left=0.07, right=0.9, top=1 - 1.2 / fig.get_figheight(), bottom=0.7 / fig.get_figheight(), hspace=0.75)
if image is not None:
	cax = fig.add_axes([0.92, 0.7 / fig.get_figheight(), 0.012, 1 - 1.9 / fig.get_figheight()])
	bar = fig.colorbar(image, cax=cax)
	bar.set_label("% busy", color=MUTED)
	bar.outline.set_visible(False)
	bar.ax.tick_params(colors=MUTED, length=0)
out2 = csv_path.with_name(csv_path.stem + "_cores.png")
fig.savefig(out2, dpi=200, facecolor=SURFACE)
print(f"Saved {out2}")
