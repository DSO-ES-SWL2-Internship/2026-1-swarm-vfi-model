# Codec/scaling performance suite for the iMX8: every combination of
#   decode {cpu, vpu}  ->  scale to 448x256 grey {cpu, g2d}  ->  encode {cpu, vpu}
# at each source resolution, repeated N times. Measures wall time (-> fps) and CPU time.
# Runs on the board with the standard library only; plot the CSV on the VM with plot_codec_suite.py.
#
# Usage: python3 codec_suite.py [--repeats 3] [--resolutions 480p 720p 1080p] [--only "vpu -> g2d"] [--keep-output]
#
# Caveat: the CPU encoder is avenc_mpeg4 (MPEG-4 Part 2) -- the image has no CPU H.264 encoder --
# so CPU-vs-VPU encode comparisons understate the VPU's advantage.

import argparse
import csv
import itertools
import resource
import shlex
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR.parent.parent / "videos"
RESULTS_DIR = VIDEOS_DIR / "benchmark"

FRAMES = 1251  # every sintel_trailer-*.mp4 source has 1251 video frames

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


def build_pipeline(video, dec, scale, enc, output_path):
	encoder, parser = ENCODERS[enc]
	# Grey -> RGB -> I420 before the encoder: a direct GRAY8 -> I420 copies full-range values into a
	# plane players read as video range. I420 is forced so vpuenc_h264 never gets RGB to convert itself.
	sink = f"{parser} ! mp4mux ! filesink location={output_path}" if output_path else "fakesink"
	return (f"filesrc location={video} ! qtdemux ! h264parse ! {DECODERS[dec]} ! {SCALERS[scale]} "
	        f"! videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=I420 ! {encoder} ! {sink}")


def run_once(pipeline):
	# getrusage(RUSAGE_CHILDREN) only covers children that have finished, so the difference
	# before/after one blocking run is exactly that gst-launch process's CPU time, all threads.
	before = resource.getrusage(resource.RUSAGE_CHILDREN)
	start = time.perf_counter()
	result = subprocess.run(["gst-launch-1.0", "-q", "-e", *shlex.split(pipeline)],
	                        capture_output=True, text=True)
	wall = time.perf_counter() - start
	after = resource.getrusage(resource.RUSAGE_CHILDREN)
	return {
		"ok": result.returncode == 0,
		"wall_s": wall,
		"cpu_user_s": after.ru_utime - before.ru_utime,
		"cpu_sys_s": after.ru_stime - before.ru_stime,
		"error": result.stderr.strip().splitlines()[-1] if result.returncode and result.stderr.strip() else "",
	}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--resolutions", nargs="+", default=["480p", "720p", "1080p"])
parser.add_argument("--only", default="",
                    help='only run configs whose name contains this text, e.g. "vpu -> g2d"')
parser.add_argument("--keep-output", action="store_true",
                    help="write each run's result to an MP4 in /tmp (for correctness checks) instead of discarding it")
args = parser.parse_args()

configs = list(itertools.product(DECODERS, SCALERS, ENCODERS))  # 8 (decoder, scaler, encoder) combos
configs = [c for c in configs if args.only in " -> ".join(c)]
if not configs:
	raise SystemExit(f"no config matches --only {args.only!r}")
videos = {}
for res in args.resolutions:
	path = VIDEOS_DIR / f"sintel_trailer-{res}.mp4"
	if path.exists():
		videos[res] = path
	else:
		print(f"skipping {res}: {path} not found")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
csv_path = RESULTS_DIR / f"codec_suite_{stamp}.csv"
summary_path = RESULTS_DIR / f"codec_suite_{stamp}.txt"

# Execute runs and print out per-run results
total = args.repeats * len(videos) * len(configs)
rows, n = [], 0
# Repeats are the OUTER loop: each pass visits every config once, so slow drift over the session
# (e.g. thermal throttling) is spread across all configs instead of landing on the last ones.
for repeat in range(1, args.repeats + 1):
	for res, video in videos.items():
		for dec, scale, enc in configs:
			n += 1
			config = f"{dec} -> {scale} -> {enc}"
			output = f"/tmp/suite_{res}_{dec}_{scale}_{enc}.mp4" if args.keep_output else None
			m = run_once(build_pipeline(video, dec, scale, enc, output))
			cpu = m["cpu_user_s"] + m["cpu_sys_s"]
			row = {"resolution": res, "decoder": dec, "scaler": scale, "encoder": enc, "config": config,
			       "repeat": repeat, "status": "ok" if m["ok"] else "failed",
			       "wall_s": round(m["wall_s"], 3), "cpu_s": round(cpu, 3),
			       "cpu_user_s": round(m["cpu_user_s"], 3), "cpu_sys_s": round(m["cpu_sys_s"], 3),
			       "fps": round(FRAMES / m["wall_s"], 1) if m["ok"] else "", "error": m["error"]}
			rows.append(row)
			result = f"{row['fps']:6.1f} fps  {cpu:6.1f} CPU-s" if m["ok"] else f"FAILED: {m['error']}"
			print(f"[{n:3d}/{total}] repeat {repeat}  {res:>5}  {config:<22} {result}", flush=True)

# Write results to a csv file
with open(csv_path, "w", newline="") as f:
	writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
	writer.writeheader()
	writer.writerows(rows)

# Print and write summary: mean (and stdev when repeated) per resolution x config, successful runs only.
lines = [f"Codec suite {stamp}: {args.repeats} repeat(s), {FRAMES} frames per run, flat out.",
         "CPU encoder = avenc_mpeg4 (MPEG-4 Part 2); VPU encoder = vpuenc_h264 (H.264).", "",
         f"{'res':>5}  {'decode -> scale -> encode':<24} {'fps mean':>9} {'sd':>6} {'CPU-s mean':>11} {'sd':>6}  runs"]
for res in videos:
	for dec, scale, enc in configs:
		config = f"{dec} -> {scale} -> {enc}"
		ok = [r for r in rows if r["resolution"] == res and r["config"] == config and r["status"] == "ok"]
		if not ok:
			lines.append(f"{res:>5}  {config:<24} {'FAILED':>9}")
			continue
		fps = [r["fps"] for r in ok]
		cpu = [r["cpu_s"] for r in ok]
		sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
		lines.append(f"{res:>5}  {config:<24} {statistics.mean(fps):9.1f} {sd(fps):6.1f} "
		             f"{statistics.mean(cpu):11.1f} {sd(cpu):6.1f}  {len(ok)}/{args.repeats}")
	lines.append("")
summary_path.write_text("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"Raw results: {csv_path}\nSummary:     {summary_path}")
