# Plots the single-layer CPU/NPU/GPU benchmark (gpu_layer_test.py models run through
# benchmark_backends.sh) as grouped horizontal bars on a log axis: one group per U-Net layer type,
# with bars grouped by unit (CPU, NPU, GPU) and both quantisation variants for the NPU and GPU.
# Usage (VM, needs matplotlib): python3 plot_gpu_layers.py <results.txt> <out.png>

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

LAYERS = [  # (file prefix, label), top to bottom
	("conv_in", "Conv2D, input\n(2→16 ch)"),
	("conv_full", "Conv2D, full res\n(32→16 ch)"),
	("convT", "Conv2DTranspose\n(upscale ×2)"),
	("concat", "Concatenate\n(skip connection)"),
	("maxpool", "MaxPooling2D"),
	("upsample_nearest", "UpSampling2D\n(alternative, not in U-Net)"),
]
# (unit, column, variants) -- bars are grouped by unit, with a wider gap between units. The CPU times
# for the two quantisation variants match to within 1%, so the CPU shows one bar.
UNITS = [
	("CPU", 0, ["perchannel"]),
	("NPU", 1, ["perchannel", "pertensor"]),
	("GPU", 2, ["perchannel", "pertensor"]),
]
# Unit colours match the report's hardware palette; per-tensor is a lighter shade of the same hue.
COLOURS = {
	("CPU", "perchannel"): "#8a8a8a",
	("NPU", "perchannel"): "#eb6834", ("NPU", "pertensor"): "#f5b08f",
	("GPU", "perchannel"): "#7b4fc9", ("GPU", "pertensor"): "#bda5e6",
}
VARIANT_NAME = {"perchannel": "per-channel", "pertensor": "per-tensor"}
TEXT, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#f6f5f2"

rows = {}
for line in open(sys.argv[1]).read().splitlines()[1:]:
	parts = line.split()
	if len(parts) >= 4:
		rows[parts[0]] = [float(v) for v in parts[1:4]]

# Offsets of each bar within a layer's slot: bars in a unit sit close together, units are spaced apart.
bar_h, bar_gap, unit_gap = 0.13, 0.015, 0.07
offsets, y = [], 0.0
for k, (unit, col, variants) in enumerate(UNITS):
	if k:
		y += unit_gap
	for variant in variants:
		offsets.append((unit, col, variant, y))
		y += bar_h + bar_gap
span = y - bar_gap - bar_h
offsets = [(u, c, v, o - span / 2) for u, c, v, o in offsets]

plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10, 7.4), facecolor="white")
for i, (prefix, _) in enumerate(LAYERS):
	if i % 2 == 0:
		ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, linewidth=0)
	for unit, col, variant, off in offsets:
		ms = rows[f"{prefix}_{variant}"][col]
		label = unit if unit == "CPU" else f"{unit}, {VARIANT_NAME[variant]}"
		ax.barh(i + off, ms, height=bar_h, color=COLOURS[(unit, variant)],
			label=label if i == 0 else None, zorder=3)
		ax.text(ms * 1.12, i + off, f"{ms:,.0f}" if ms >= 10 else f"{ms:.1f}",
			va="center", fontsize=8, color=MUTED, zorder=4)
ax.set_xscale("log")
ax.set_xlim(1, 80000)
ticks = [1, 10, 100, 1000, 10000]
ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(len(LAYERS) - 0.5, -0.5)
ax.set_yticks(range(len(LAYERS)), [label for _, label in LAYERS], color=TEXT)
ax.set_xlabel("milliseconds per inference (log scale, shorter is faster)", color=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
for side in ["top", "right", "left"]:
	ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUTED, length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5, frameon=False, fontsize=9.5,
	handlelength=1.2, columnspacing=1.4)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=200, facecolor="white")
print(f"Saved {sys.argv[2]}")
