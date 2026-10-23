# How does moving each step of decode -> scale -> encode onto hardware, and putting FIFOs (queue elements)
# between the steps, change the speed? Each configuration runs the 1080p trailer flat out (fakesink sync=false)
# and reports the wall time per frame and the CPU time it used.
#
# Run on the iMX8: python3 streaming/scripts/test/pipeline_suite.py [--group steps|avdec] [--repeats 2] [--res 1080p]
#
# Without queues, one GStreamer thread runs every step for a frame before starting the next frame, so the
# time per frame is about the SUM of the steps. A queue starts a new thread for the steps after it, so with a
# queue between every pair of steps they work on different frames at once and the time per frame is about
# the SLOWEST step. avdec_h264 is the exception: it decodes on its own worker threads, which already overlap
# with the steps after it (an implicit FIFO).
#
# CPU time is the whole board's busy time during the run (/proc/stat, same as codec_suite.py), so it
# includes driver work for the hardware units. "cores" = CPU time / wall time = average cores busy (of 4).
#
# Format glue: the G2D outputs RGB only. vpuenc_h264 takes RGBx directly; x264enc doesn't, so the G2D -> CPU
# encode configurations add a CPU videoconvert to I420, counted as part of the scale step.
#
# The last two rows are the relay's real video chain (relay.py, HW mode, --range rgb) without the model:
# G2D RGB -> grey for the model (CPU videoconvert), then grey -> RGBx -> I420 for the encoder (CPU), each
# conversion a step of its own, so with FIFOs every one gets its own thread.
#
# --group avdec tests the claim that avdec_h264 already overlaps with the steps after it: CPU decode -> G2D
# scale -> VPU encode, with and without FIFOs, so the only CPU-heavy step is the decode itself. "cpu1" is the
# control, avdec_h264 limited to one thread: with no worker threads to overlap, the FIFOs should help it the way
# they help vpudec.

import argparse
import csv
import subprocess
import time
from pathlib import Path

VIDEOS = Path(__file__).resolve().parent.parent.parent / "videos"
OUT = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "pipeline_suite"

DECODE = {"cpu": "avdec_h264", "cpu1": "avdec_h264 max-threads=1", "vpu": "vpudec frame-drop=false"}
SCALE = {"cpu": "videoscale ! video/x-raw,width=448,height=256",
         "g2d": "imxvideoconvert_g2d ! video/x-raw,width=448,height=256"}
ENCODE = {"cpu": "x264enc bitrate=2000", "vpu": "vpuenc_h264 bitrate=2000"}
RELAY_CONVERSIONS = [  # relay.py INPUT_CONVERTERS["g2d"] and make_output_converter()
	"videoconvert ! video/x-raw,format=GRAY8,colorimetry=2:4:0:0",
	"videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=I420",
]

# (decode, scale, encode, queues[, relay conversions]), in the order of the slides. None = step left out.
CONFIGS = {"steps": [
	("cpu", None, None, False),  # decode alone, for reference
	("vpu", None, None, False),
	("cpu", "cpu", "cpu", False),  # 1. all CPU
	("vpu", "cpu", "cpu", False),  # 2. VPU decode
	("vpu", "g2d", "cpu", False),  # 3. + G2D scale
	("vpu", "cpu", "cpu", True),  # 4. + FIFOs
	("vpu", "g2d", "cpu", True),
	("vpu", "g2d", "vpu", False),  # 5. + VPU encode
	("vpu", "g2d", "vpu", True),
	("vpu", "cpu", "vpu", True),
	("cpu", "cpu", "cpu", True),  # control: FIFOs on the all-CPU pipeline
	("vpu", "g2d", "vpu", False, True),  # 6. the relay's real chain, with its colour conversions
	("vpu", "g2d", "vpu", True, True),
], "avdec": [
	("cpu", None, None, False),  # decode alone: the slowest step
	("cpu", "g2d", "vpu", False),
	("cpu", "g2d", "vpu", True),
	("cpu1", None, None, False),  # control: avdec_h264 on one thread
	("cpu1", "g2d", "vpu", False),
	("cpu1", "g2d", "vpu", True),
]}


def label(dec, scale, enc, queues, convert=False):
	steps = [f"{dec} dec"] + ([f"{scale} scale"] if scale else []) + (["grey", "I420"] if convert else []) + \
	        ([f"{enc} enc"] if enc else [])
	return (" -> fifo -> " if queues else " -> ").join(steps)


def pipeline(video, dec, scale, enc, queues, convert=False):
	steps = [DECODE[dec]]
	if scale:
		glue = scale == "g2d" and enc == "cpu" and not convert
		steps.append(SCALE[scale] + (" ! videoconvert ! video/x-raw,format=I420" if glue else ""))
	if convert:
		steps += RELAY_CONVERSIONS
	if enc:
		steps.append(ENCODE[enc])
	link = " ! queue ! " if queues else " ! "
	return f"filesrc location={video} ! qtdemux ! h264parse ! " + link.join(steps) + " ! fakesink sync=false"


def cpu_busy():
	# Busy jiffies of all cores together (user + nice + system + irq + softirq), in seconds.
	v = [int(x) for x in open("/proc/stat").readline().split()[1:8]]
	return (v[0] + v[1] + v[2] + v[5] + v[6]) / 100


def run(desc):
	busy, start = cpu_busy(), time.monotonic()
	subprocess.run(["gst-launch-1.0", "-q", *desc.split()], check=True, stdout=subprocess.DEVNULL)
	return time.monotonic() - start, cpu_busy() - busy


def main():
	parser = argparse.ArgumentParser(description="decode -> scale -> encode with and without FIFOs")
	parser.add_argument("--repeats", type=int, default=2, help="runs per configuration; the fastest counts")
	parser.add_argument("--res", default="1080p", choices=["480p", "720p", "1080p"])
	parser.add_argument("--frames", type=int, default=1251, help="frames in the video")
	parser.add_argument("--group", default="steps", choices=CONFIGS, help="steps: the slides' sequence; "
	                    "avdec: CPU decode with and without FIFOs")
	args = parser.parse_args()
	video = VIDEOS / f"sintel_trailer-{args.res}.mp4"

	OUT.mkdir(parents=True, exist_ok=True)
	out = OUT / f"pipeline_suite_{args.group}_{args.res}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
	print(f"{args.res}, {args.frames} frames, fastest of {args.repeats}")
	print(f"{'configuration':58} {'wall s':>7} {'ms/frame':>9} {'fps':>6} {'CPU s':>6} {'cores':>6}")
	with open(out, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["configuration", "decode", "scale", "encode", "fifos", "wall_s", "ms_per_frame", "fps",
		                 "cpu_s", "cores_busy", "all_runs_wall_s"])
		for config in CONFIGS[args.group]:
			runs = [run(pipeline(video, *config)) for _ in range(args.repeats)]
			wall, cpu = min(runs)
			name = label(*config)
			dec, scale, enc, queues = config[:4]
			ms, fps = wall / args.frames * 1000, args.frames / wall
			print(f"{name:58} {wall:7.2f} {ms:9.2f} {fps:6.1f} {cpu:6.2f} {cpu / wall:6.2f}", flush=True)
			writer.writerow([name, dec, scale or "", enc or "", queues, f"{wall:.3f}", f"{ms:.2f}", f"{fps:.1f}",
			                 f"{cpu:.2f}", f"{cpu / wall:.2f}", " ".join(f"{w:.2f}" for w, _ in runs)])
	print(f"saved {out}")


if __name__ == "__main__":
	main()
