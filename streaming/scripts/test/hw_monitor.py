# Live load of every accelerator on the iMX8, one screen, refreshed every second: CPU (per core),
# GPU and NPU (galcore's gc/load), and VPU decode/encode rate (interrupt counts). Runs beside any
# pipeline -- e.g. the real relay in Demo 2 -- instead of nnshark, whose live tracer breaks the relay's
# TCP output. Standard library only.
#
# Usage: python3 hw_monitor.py [--interval 1] [--csv load.csv]
#   gc/load needs debugfs: mount -t debugfs none /sys/kernel/debug
#
# What each number is:
#   CPU   busy % per core from /proc/stat (the counters top/htop read); includes this script (~1%).
#   GPU   gc/load core 0, NPU = core 1: the galcore driver's busy %.
#   VPU   no busy % exists; hantrodec (G1 decoder) and hx280enc (VC8000E encoder) interrupts per
#         second, divided by IRQS_PER_FRAME = frames per second.
#   G2D   has no load counter at all (gc/load stays at 0% while it scales), so it isn't shown.

import argparse
import csv
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

GC_LOAD_PATH = Path("/sys/kernel/debug/gc/load")
VPU_IRQS = {"decode": "hantrodec", "encode": "hx280enc"}
# Measured on the board (20 Oct): one nnshark_demo.sh 480p run (1251 frames) raised exactly 1251
# hantrodec and 1251 hx280enc interrupts.
IRQS_PER_FRAME = 1
VPU_SCALE = 60  # frames/s that fills the VPU bars (48 = Demo 2's output rate)
HISTORY = 30  # seconds of sparkline per row

BAR_W = 24
SPARKS = " ▁▂▃▄▅▆▇█"
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def read_cpu_ticks():
	# Per-core (busy, total) jiffies; idle and iowait count as not busy (same as codec_suite.py).
	ticks = []
	with open("/proc/stat") as f:
		for line in f:
			if re.match(r"cpu\d+ ", line):
				v = [int(x) for x in line.split()[1:9]]
				ticks.append((sum(v) - v[3] - v[4], sum(v)))
	return ticks


def read_gc_load():
	# {core: load%}, or None where gc/load isn't readable (debugfs not mounted, or not on the board).
	try:
		text = GC_LOAD_PATH.read_text()
	except OSError:
		return None
	return {int(c): int(l) for c, l in re.findall(r"core\s*:\s*(\d+)\s*\n\s*load\s*:\s*(\d+)%", text)}


def read_vpu_irqs():
	# {"decode": count, "encode": count}, summed over all cores and over every line with that driver
	# name (hantrodec has two: G1 and the unused G2). Missing driver = no key.
	counts = {}
	with open("/proc/interrupts") as f:
		for line in f:
			fields = line.split()
			for unit, name in VPU_IRQS.items():
				if fields and fields[-1] == name:
					counts[unit] = counts.get(unit, 0) + sum(int(x) for x in fields[1:] if x.isdigit())
	return counts


def colour(pct):
	return GREEN if pct < 50 else YELLOW if pct < 80 else RED


def bar(pct, width=BAR_W):
	pct = max(0.0, min(100.0, pct))
	filled = round(pct / 100 * width)
	return f"{colour(pct)}{'█' * filled}{RESET}{DIM}{'·' * (width - filled)}{RESET}"


def spark(history):
	return DIM + "".join(SPARKS[min(8, round(max(0.0, min(100.0, p)) / 100 * 8))] for p in history) + RESET


def row(label, pct, history, value):
	return f"  {BOLD}{label:<7}{RESET} {bar(pct)}  {value:>13}  {spark(history)}"


def draw(stamp, cores, gc, vpu, hist):
	total = sum(cores) / len(cores)
	lines = [f"  {BOLD}{CYAN}i.MX 8M Plus · hardware load{RESET}{DIM}   {stamp}   (Ctrl-C to stop){RESET}", ""]
	lines.append(row("CPU", total, hist["cpu"], f"{total:5.0f}% of all"))
	for i, pct in enumerate(cores):
		lines.append(row(f" core {i}", pct, hist[f"core{i}"], f"{pct:5.0f}%"))
	lines.append("")
	if gc is None:
		lines.append(f"  {BOLD}GPU/NPU{RESET} {DIM}gc/load not readable -- mount -t debugfs none /sys/kernel/debug{RESET}")
	else:
		lines.append(row("GPU", gc.get(0, 0), hist["gpu"], f"{gc.get(0, 0):5.0f}%"))
		lines.append(row("NPU", gc.get(1, 0), hist["npu"], f"{gc.get(1, 0):5.0f}%"))
	lines.append("")
	for unit in VPU_IRQS:
		if unit in vpu:
			fps = vpu[unit]
			lines.append(row(f"VPU {unit[:3]}", 100 * fps / VPU_SCALE, hist[unit], f"{fps:5.0f} frames/s"))
		else:
			lines.append(f"  {BOLD}VPU {unit[:3]}{RESET} {DIM}no {VPU_IRQS[unit]} interrupt line{RESET}")
	lines.append("")
	lines.append(f"  {DIM}G2D: no load counter. VPU bars: {VPU_SCALE} frames/s = full bar; rate, not busy %.{RESET}")
	# Home the cursor and clear below, so the screen updates in place without flicker.
	sys.stdout.write("\033[H" + "\n".join(line + "\033[K" for line in lines) + "\033[J")
	sys.stdout.flush()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--interval", type=float, default=1.0, help="seconds between updates")
	parser.add_argument("--csv", help="also append every sample to this CSV file")
	args = parser.parse_args()
	sys.stdout.reconfigure(encoding="utf-8")  # block characters, even under the board's POSIX locale

	prev_cpu, prev_irq, prev_t = read_cpu_ticks(), read_vpu_irqs(), time.monotonic()
	hist = {}
	writer = None
	if args.csv:
		csv_file = open(args.csv, "a", newline="")
		writer = csv.writer(csv_file)
		if csv_file.tell() == 0:
			writer.writerow(["time", "cpu_total_pct", *[f"core{i}_pct" for i in range(len(prev_cpu))],
			                 "gpu_pct", "npu_pct", "vpu_decode_fps", "vpu_encode_fps"])

	sys.stdout.write("\033[?25l\033[2J")  # hide the cursor, clear the screen
	try:
		while True:
			time.sleep(args.interval)
			cpu, irq, now = read_cpu_ticks(), read_vpu_irqs(), time.monotonic()
			cores = [100 * (cb - pb) / (ct - pt) if ct > pt else 0.0 for (pb, pt), (cb, ct) in zip(prev_cpu, cpu)]
			dt = now - prev_t
			vpu = {u: (irq[u] - prev_irq.get(u, irq[u])) / IRQS_PER_FRAME / dt for u in irq}
			gc = read_gc_load()
			prev_cpu, prev_irq, prev_t = cpu, irq, now

			values = {"cpu": sum(cores) / len(cores), **{f"core{i}": p for i, p in enumerate(cores)}}
			if gc is not None:
				values.update(gpu=gc.get(0, 0), npu=gc.get(1, 0))
			values.update({u: 100 * fps / VPU_SCALE for u, fps in vpu.items()})
			for key, pct in values.items():
				hist.setdefault(key, deque(maxlen=HISTORY)).append(pct)

			stamp = datetime.now().strftime("%H:%M:%S")
			draw(stamp, cores, gc, vpu, hist)
			if writer:
				writer.writerow([stamp, f"{values['cpu']:.1f}", *[f"{p:.1f}" for p in cores],
				                 gc.get(0, "") if gc else "", gc.get(1, "") if gc else "",
				                 *[f"{vpu[u]:.1f}" if u in vpu else "" for u in VPU_IRQS]])
				csv_file.flush()
	except KeyboardInterrupt:
		pass
	finally:
		sys.stdout.write("\033[?25h\n")  # show the cursor again


if __name__ == "__main__":
	main()
