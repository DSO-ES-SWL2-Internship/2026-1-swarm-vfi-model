# Per-step CPU cost of the codec suite's pipelines: builds each pipeline up one stage at a time
# (decode, + scale, + grey, + grey -> RGBx, + RGBx -> I420, + encode), runs each flat out, and reports
# the CPU time each added stage costs per frame. Finds which colour conversion makes the colour-correct
# pipeline ~6.5 ms/frame more expensive than the original one. Also times some alternatives
# (direct GRAY8 -> I420, multithreaded videoconvert) against the full pipeline.
# Runs on the board with the standard library only.
#
# Usage: python3 conversion_cost.py [--resolution 480p] [--repeats 3]

import argparse
import resource
import shlex
import statistics
import subprocess
import time
from pathlib import Path

VIDEOS_DIR = Path(__file__).resolve().parent.parent.parent / "videos"
FRAMES = 1251

SIZE = "video/x-raw,width=448,height=256"
GREY = "videoconvert ! video/x-raw,format=GRAY8,colorimetry=2:4:0:0"  # as in codec_suite's g2d scaler
TO_RGBX = "videoconvert ! video/x-raw,format=RGBx"
TO_I420 = "videoconvert ! video/x-raw,format=I420"
ENCODE = "vpuenc_h264 bitrate=2000"

# Each list adds one stage to the previous entry; the difference between neighbours is that stage's cost.
STEPS = {
	"vpu": [
		("decode (vpudec)", "vpudec frame-drop=false"),
		("+ scale (G2D)", f"imxvideoconvert_g2d ! {SIZE}"),
		("+ grey (RGB -> GRAY8)", GREY),
		("+ GRAY8 -> RGBx", TO_RGBX),
		("+ RGBx -> I420", TO_I420),
		("+ encode (vpuenc_h264)", ENCODE),
	],
	"cpu": [
		("decode (avdec_h264)", "avdec_h264"),
		("+ scale (videoscale)", f"videoscale ! {SIZE}"),
		("+ I420 -> RGBx", TO_RGBX),
		("+ RGBx -> GRAY8", "videoconvert ! video/x-raw,format=GRAY8"),
		("+ GRAY8 -> RGBx", TO_RGBX),
		("+ RGBx -> I420", TO_I420),
		("+ encode (vpuenc_h264)", ENCODE),
	],
}
# Whole-pipeline alternatives for the VPU path, compared with its full pipeline above.
ALTERNATIVES = [
	# Wrong range (copies full-range grey into a video-range plane); shows what the RGBx detour costs.
	("alt: GRAY8 -> I420 directly", ["vpudec frame-drop=false", f"imxvideoconvert_g2d ! {SIZE}", GREY, TO_I420, ENCODE]),
	# Same conversions, each videoconvert split over all 4 cores.
	("alt: videoconvert n-threads=4", ["vpudec frame-drop=false", f"imxvideoconvert_g2d ! {SIZE}",
	                                   GREY.replace("videoconvert", "videoconvert n-threads=4"),
	                                   TO_RGBX.replace("videoconvert", "videoconvert n-threads=4"),
	                                   TO_I420.replace("videoconvert", "videoconvert n-threads=4"), ENCODE]),
]


def run(video, stages):
	pipeline = f"filesrc location={video} ! qtdemux ! h264parse ! {' ! '.join(stages)} ! fakesink sync=false"
	before = resource.getrusage(resource.RUSAGE_CHILDREN)
	start = time.perf_counter()
	proc = subprocess.run(["gst-launch-1.0", "-q", *shlex.split(pipeline)], capture_output=True, text=True)
	wall = time.perf_counter() - start
	after = resource.getrusage(resource.RUSAGE_CHILDREN)
	if proc.returncode:
		lines = (proc.stderr or proc.stdout).splitlines()
		raise SystemExit(f"pipeline failed: {pipeline}\n" + "\n".join(lines[:5]))
	cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
	return cpu, wall


parser = argparse.ArgumentParser()
parser.add_argument("--resolution", default="480p")
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
video = VIDEOS_DIR / f"sintel_trailer-{args.resolution}.mp4"

# (label, stage list): the cumulative steps of each path, then the alternatives
cases = []
for path, steps in STEPS.items():
	for i in range(len(steps)):
		cases.append((path, steps[i][0], [s for _, s in steps[:i + 1]]))
cases += [("vpu", label, stages) for label, stages in ALTERNATIVES]

print(f"{video.name}, {FRAMES} frames, flat out, mean of {args.repeats} runs. "
      "CPU ms/frame = user + system CPU time / frames.\n")
print(f"{'path':<5} {'step':<32} {'fps':>6} {'CPU ms/frame':>13} {'step cost':>10}")
prev = {}
for path, label, stages in cases:
	results = [run(video, stages) for _ in range(args.repeats)]
	cpu_ms = statistics.mean(c for c, _ in results) / FRAMES * 1000
	fps = FRAMES / statistics.mean(w for _, w in results)
	if label.startswith("alt"):
		step = f"{cpu_ms - prev['vpu_full']:+.2f}"  # vs the full VPU pipeline
	elif label.startswith("+"):
		step = f"{cpu_ms - prev[path]:+.2f}"
	else:
		step = ""
	print(f"{path:<5} {label:<32} {fps:6.1f} {cpu_ms:13.2f} {step:>10}", flush=True)
	prev[path] = cpu_ms
	if path == "vpu" and label == STEPS["vpu"][-1][0]:
		prev["vpu_full"] = cpu_ms
print("\nstep cost = extra CPU ms/frame over the line above (alternatives: over the full VPU pipeline)")
