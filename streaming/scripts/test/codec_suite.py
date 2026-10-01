# Codec/scaling performance suite for the iMX8: every combination of
#   decode {cpu, vpu}  ->  scale to 448x256 grey {cpu, g2d}  ->  encode {cpu, vpu}
# at each source resolution, repeated N times. Measures wall time (-> fps), CPU time charged to the
# pipeline, and htop-style utilisation (system-wide, per core, over time) plus GPU/NPU load.
# Runs on the board with the standard library only; plot the CSV on the VM with plot_codec_suite.py.
#
# Usage: python3 codec_suite.py [--repeats 3] [--resolutions 480p 720p 1080p] [--only "vpu -> g2d"]
#                               [--pace 60] [--keep-output]
#   --pace 60 plays the 60 fps copies of the sources in real time instead of flat out, so every config
#   does the same work per second -- utilisation then shows how much CPU each needs to keep up.
#
# Caveat: the CPU encoder is avenc_mpeg4 (MPEG-4 Part 2) -- the image has no CPU H.264 encoder --
# so CPU-vs-VPU encode comparisons understate the VPU's advantage.

import argparse
import csv
import itertools
import os
import re
import resource
import shlex
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR.parent.parent / "videos"
RESULTS_DIR = VIDEOS_DIR / "benchmark"

FRAMES = 1251  # every sintel_trailer-*.mp4 source has 1251 video frames
SAMPLE_INTERVAL_S = 0.1
GC_LOAD_PATH = Path("/sys/kernel/debug/gc/load")  # galcore driver: core 0 = GPU, core 1 = NPU
N_CORES = os.cpu_count()

DECODERS = {
	"cpu": "avdec_h264",
	# frame-drop defaults to true: it would skip frames when busy, making VPU's numbers better than is fair.
	"vpu": "vpudec frame-drop=false",
}
SCALERS = {
	# Through RGB -> full-range grey (0-255), matching training (PIL convert("L")); a direct YUV -> GRAY8
	# copies the video-range (16-235) luma plane instead. Scaling first keeps that extra step cheap.
	"cpu": "videoscale ! video/x-raw,width=448,height=256 ! videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=GRAY8",
	# G2D resizes (output is RGB only); grey conversion afterwards is on the CPU at 448x256. The GRAY8
	# colorimetry was chosen by measurement: it is the setting whose output matches the training convention.
	"g2d": "imxvideoconvert_g2d ! video/x-raw,width=448,height=256 ! videoconvert ! video/x-raw,format=GRAY8,colorimetry=2:4:0:0",
}
ENCODERS = {  # (encoder, parser needed to mux its output into MP4 for --keep-output)
	"cpu": ("avenc_mpeg4 bitrate=2000000", "mpeg4videoparse"),
	"vpu": ("vpuenc_h264 bitrate=2000", "h264parse"),
}


def build_pipeline(video, dec, scale, enc, output_path, pace):
	encoder, parser = ENCODERS[enc]
	# sync=true consumes each frame at its timestamp (real-time pacing); sync=false runs flat out.
	sync = "true" if pace else "false"
	# The 60 fps copies' frames are exactly 1/60 s apart, but qtdemux estimates the caps' rate from the
	# file duration and reports 60000/1001, which vpuenc_h264 rejects ("VCEncCheckCfg: Invalid
	# frameRateNum"). capssetter only rewrites the declared rate -- no per-frame work.
	rate = f' ! capssetter caps="video/x-h264,framerate={pace}/1"' if pace else ""
	# Grey -> RGB -> I420 before the encoder: a direct GRAY8 -> I420 copies full-range values into a
	# plane players read as video range. I420 is forced so vpuenc_h264 never gets RGB to convert itself.
	sink = (f"{parser} ! mp4mux ! filesink sync={sync} location={output_path}" if output_path
	        else f"fakesink sync={sync}")
	return (f"filesrc location={video} ! qtdemux ! h264parse{rate} ! {DECODERS[dec]} ! {SCALERS[scale]} "
	        f"! videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=I420 ! {encoder} ! {sink}")


def read_cpu_ticks():
	# Per-core (busy, total) jiffies from /proc/stat -- the same counters htop reads. idle and iowait
	# count as not busy; everything else (user, nice, system, irq, softirq, steal) counts as busy.
	ticks = []
	with open("/proc/stat") as f:
		for line in f:
			if re.match(r"cpu\d+ ", line):
				v = [int(x) for x in line.split()[1:9]]
				ticks.append((sum(v) - v[3] - v[4], sum(v)))
	return ticks


def read_gc_load():
	# {core: load%} from the galcore driver, or {} where it isn't readable (e.g. on the VM).
	try:
		text = GC_LOAD_PATH.read_text()
	except OSError:
		return {}
	return {int(c): int(l) for c, l in re.findall(r"core\s*:\s*(\d+)\s*\n\s*load\s*:\s*(\d+)%", text)}


def run_once(pipeline):
	# getrusage(RUSAGE_CHILDREN) only covers children that have been waited for, so the difference
	# before/after one run is exactly that gst-launch process's CPU time, all threads. The /proc/stat
	# samples are system-wide instead: they also catch work not charged to the pipeline (kernel threads,
	# interrupts from the VPU/G2D drivers), at the cost of also counting this sampler (~1% of one core).
	before = resource.getrusage(resource.RUSAGE_CHILDREN)
	samples = []
	with tempfile.TemporaryFile(mode="w+") as err:
		start = time.perf_counter()
		proc = subprocess.Popen(["gst-launch-1.0", "-q", "-e", *shlex.split(pipeline)],
		                        stdout=subprocess.DEVNULL, stderr=err, text=True)
		prev = read_cpu_ticks()
		while proc.poll() is None:
			time.sleep(SAMPLE_INTERVAL_S)
			cur = read_cpu_ticks()
			per_core = [100 * (cb - pb) / (ct - pt) if ct > pt else 0.0 for (pb, pt), (cb, ct) in zip(prev, cur)]
			samples.append({"t": time.perf_counter() - start, "cores": per_core, "gc": read_gc_load()})
			prev = cur
		wall = time.perf_counter() - start
		err.seek(0)
		stderr = err.read().strip()
	after = resource.getrusage(resource.RUSAGE_CHILDREN)
	return {
		"ok": proc.returncode == 0,
		"wall_s": wall,
		"cpu_user_s": after.ru_utime - before.ru_utime,
		"cpu_sys_s": after.ru_stime - before.ru_stime,
		"samples": samples,
		"error": first_error(stderr) if proc.returncode else "",
	}


def first_error(stderr):
	# gst-launch's last line is often a generic follow-up ("pipeline doesn't want to preroll"); the
	# cause is the first "ERROR: from element ..." line and the debug info printed after it.
	lines = stderr.splitlines()
	start = next((i for i, line in enumerate(lines) if line.startswith("ERROR")), 0)
	return " | ".join(line.strip() for line in lines[start:start + 3])


STARTUP_S = 1.0  # left out of the summary: plugin loading spikes the CPU at the start of every run


def utilisation(samples):
	# Summary of the samples: total CPU as % of all cores (100 = every core fully busy), its 95th
	# percentile (robust "near-peak": /proc/stat counts 10 ms ticks, so a single 0.1 s sample is noisy),
	# the per-core means, and mean GPU/NPU load when the driver reports it. The first STARTUP_S seconds
	# are excluded (they stay in the time-series file) so start-up doesn't mask differences between configs.
	samples = [s for s in samples if s["t"] >= STARTUP_S] or samples
	if not samples:
		return {}
	totals = [sum(s["cores"]) / N_CORES for s in samples]
	out = {"cpu_util_mean_pct": round(statistics.mean(totals), 1), "cpu_util_p95_pct": round(statistics.quantiles(totals, n=20)[-1] if len(totals) > 1 else totals[0], 1)}
	for core in range(N_CORES):
		out[f"cpu{core}_mean_pct"] = round(statistics.mean(s["cores"][core] for s in samples), 1)
	for core, name in ((0, "gpu"), (1, "npu")):
		loads = [s["gc"][core] for s in samples if core in s["gc"]]
		out[f"{name}_load_mean_pct"] = round(statistics.mean(loads), 1) if loads else ""
	return out


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--resolutions", nargs="+", default=["480p", "720p", "1080p"])
parser.add_argument("--only", default="",
                    help='only run configs whose name contains this text, e.g. "vpu -> g2d"')
parser.add_argument("--pace", type=int, default=0,
                    help="play the <pace>fps copies of the sources in real time (0 = flat out); needs "
                         "sintel_trailer-<res>-<pace>fps.mp4, e.g. made with ffmpeg -itsscale")
parser.add_argument("--keep-output", action="store_true",
                    help="write each run's result to an MP4 in /tmp (for correctness checks) instead of discarding it")
args = parser.parse_args()

configs = list(itertools.product(DECODERS, SCALERS, ENCODERS))  # 8 (decoder, scaler, encoder) combos
configs = [c for c in configs if args.only in " -> ".join(c)]
if not configs:
	raise SystemExit(f"no config matches --only {args.only!r}")
suffix = f"-{args.pace}fps" if args.pace else ""
videos = {}
for res in args.resolutions:
	path = VIDEOS_DIR / f"sintel_trailer-{res}{suffix}.mp4"
	if path.exists():
		videos[res] = path
	else:
		print(f"skipping {res}: {path} not found")
if not read_gc_load():
	print(f"note: {GC_LOAD_PATH} not readable -- GPU/NPU load will be blank (mount debugfs as root on the board)")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mode = f"paced{args.pace}" if args.pace else "flatout"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
csv_path = RESULTS_DIR / f"codec_suite_{mode}_{stamp}.csv"
timeseries_path = RESULTS_DIR / f"codec_suite_{mode}_{stamp}_timeseries.csv"
summary_path = RESULTS_DIR / f"codec_suite_{mode}_{stamp}.txt"
expected_s = FRAMES / args.pace if args.pace else None

# Execute runs and print out per-run results
total = args.repeats * len(videos) * len(configs)
rows, series, n = [], [], 0
# Repeats are the OUTER loop: each pass visits every config once, so slow drift over the session
# (e.g. thermal throttling) is spread across all configs instead of landing on the last ones.
for repeat in range(1, args.repeats + 1):
	for res, video in videos.items():
		for dec, scale, enc in configs:
			n += 1
			config = f"{dec} -> {scale} -> {enc}"
			output = f"/tmp/suite_{res}_{dec}_{scale}_{enc}.mp4" if args.keep_output else None
			m = run_once(build_pipeline(video, dec, scale, enc, output, args.pace))
			cpu = m["cpu_user_s"] + m["cpu_sys_s"]
			row = {"mode": mode, "resolution": res, "decoder": dec, "scaler": scale, "encoder": enc,
			       "config": config, "repeat": repeat, "status": "ok" if m["ok"] else "failed",
			       "wall_s": round(m["wall_s"], 3), "cpu_s": round(cpu, 3),
			       "cpu_user_s": round(m["cpu_user_s"], 3), "cpu_sys_s": round(m["cpu_sys_s"], 3),
			       "fps": round(FRAMES / m["wall_s"], 1) if m["ok"] else "",
			       # paced runs: how far behind real time the run finished (small = kept up; includes ~0.3s startup)
			       "lag_s": round(m["wall_s"] - expected_s, 2) if expected_s and m["ok"] else "",
			       **utilisation(m["samples"]), "error": m["error"]}
			rows.append(row)
			for s in m["samples"]:
				series.append({"resolution": res, "config": config, "repeat": repeat, "t": round(s["t"], 2),
				               **{f"cpu{c}": round(v, 1) for c, v in enumerate(s["cores"])},
				               "gpu": s["gc"].get(0, ""), "npu": s["gc"].get(1, "")})
			if m["ok"]:
				lag = f"  lag {row['lag_s']:+5.2f}s" if expected_s else ""
				result = (f"{row['fps']:6.1f} fps  {cpu:6.1f} CPU-s  CPU {row.get('cpu_util_mean_pct', 0):5.1f}% "
				          f"(p95 {row.get('cpu_util_p95_pct', 0):5.1f}%){lag}")
			else:
				result = f"FAILED: {m['error']}"
			print(f"[{n:3d}/{total}] repeat {repeat}  {res:>5}  {config:<22} {result}", flush=True)

# Write results to csv files: one row per run, and one row per utilisation sample
fields = list(dict.fromkeys(k for r in rows for k in r))  # union of keys, in first-seen order
with open(csv_path, "w", newline="") as f:
	writer = csv.DictWriter(f, fieldnames=fields, restval="")
	writer.writeheader()
	writer.writerows(rows)
if series:
	with open(timeseries_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(series[0].keys()))
		writer.writeheader()
		writer.writerows(series)

# Print and write summary: mean (and stdev when repeated) per resolution x config, successful runs only.
pacing = f"paced at {args.pace} fps (real time, {expected_s:.1f}s of video)" if args.pace else "flat out"
lines = [f"Codec suite {stamp}: {args.repeats} repeat(s), {FRAMES} frames per run, {pacing}.",
         "CPU encoder = avenc_mpeg4 (MPEG-4 Part 2); VPU encoder = vpuenc_h264 (H.264).",
         f"CPU % = system-wide busy share of all {N_CORES} cores (100% = all cores fully busy).", "",
         f"{'res':>5}  {'decode -> scale -> encode':<24} {'fps':>7} {'CPU-s':>7} {'CPU %':>6} {'p95 %':>7} "
         f"{'GPU %':>6} {'NPU %':>6} {'lag s':>6}  runs"]
for res in videos:
	for dec, scale, enc in configs:
		config = f"{dec} -> {scale} -> {enc}"
		ok = [r for r in rows if r["resolution"] == res and r["config"] == config and r["status"] == "ok"]
		if not ok:
			lines.append(f"{res:>5}  {config:<24} {'FAILED':>7}")
			continue

		def mean(key):
			vals = [r[key] for r in ok if r.get(key) not in ("", None)]
			return f"{statistics.mean(vals):.1f}" if vals else "-"
		lines.append(f"{res:>5}  {config:<24} {mean('fps'):>7} {mean('cpu_s'):>7} {mean('cpu_util_mean_pct'):>6} "
		             f"{mean('cpu_util_p95_pct'):>7} {mean('gpu_load_mean_pct'):>6} {mean('npu_load_mean_pct'):>6} "
		             f"{mean('lag_s'):>6}  {len(ok)}/{args.repeats}")
	lines.append("")
summary_path.write_text("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"Raw results: {csv_path}\nTime series: {timeseries_path}\nSummary:     {summary_path}")
