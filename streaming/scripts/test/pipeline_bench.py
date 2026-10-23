# How does moving each step of decode -> scale -> encode onto hardware, and putting FIFOs (queue elements)
# between the steps, change the speed? And can the time per frame be predicted from the steps?
#
# Run on the iMX8: python3 streaming/scripts/test/pipeline_bench.py [--repeats 2] [--res 1080p] [--match TEXT]
#
# For every configuration, the 1080p trailer runs flat out (fakesink sync=false), without and with FIFOs:
#   whole pipeline  wall time per frame and the CPU time used (fastest of --repeats runs, no probes)
#   each step       time per frame of every step (no-FIFO run with probes, see below)
#   prediction      without FIFOs: one thread runs every step of a frame, so the time per frame is the SUM of
#                   the steps. With a FIFO between every pair of steps, each step has its own thread and they
#                   work on different frames at once, so it is the SLOWEST step (as in an instruction pipeline).
#
# Step times: every step after the decoder is timed inside the real no-FIFO pipeline, on the memory it really
# gets (CPU scaling of VPU frames pays for the 4 KB pages, see scaler_tlb.py). A pad probe on each element's sink
# pad (frame enters) and src pad (frame leaves) gives the element's time per frame; with one thread nothing
# overlaps. A step is the sum of its elements (e.g. grey -> I420 is two videoconverts). The decoder can't be
# timed this way, since it doesn't send out a frame for each frame in (avdec_h264 decodes several frames ahead on
# worker threads), so the decode step is the decode-alone time per frame. The FIFO prediction uses the same step
# times.
#
# Where a prediction misses, that is the finding:
#   - avdec_h264 overlaps decoding with the later steps on its own (an implicit FIFO), so without FIFOs it beats
#     the sum. "cpu1" is the control, avdec_h264 on one thread, which should match the sum.
#   - x264enc also encodes on its own threads, so its step time is how long the pipeline's thread waits for it,
#     not its CPU work.
#   - With FIFOs, CPU steps compete for the 4 cores, so a pipeline that fills the CPU misses the max.
#
# CPU time is the whole board's busy time during the run (/proc/stat), so it includes driver work for the
# hardware units. "cores" = CPU time / wall time = average cores busy (of 4).
#
# Format glue: the G2D outputs RGB only. vpuenc_h264 takes RGBx directly; x264enc doesn't, so the G2D -> CPU
# encode configurations add a CPU videoconvert to I420, counted as part of the scale step.
#
# The relay rows are the relay's real video chain (relay.py, HW mode, --range rgb) without the model: G2D RGB ->
# grey for the model (CPU videoconvert), then grey -> RGBx -> I420 for the encoder (CPU), each conversion a step.
#
# The probes call Python twice per element per frame (~0.05 ms each), so the probed runs are a little slower;
# their wall time is saved only as a check. Each probed run is its own process (several GStreamer runs in one
# process crashed scaler_tlb.py).

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

VIDEOS = Path(__file__).resolve().parent.parent.parent / "videos"
OUT = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "pipeline_bench"
WARMUP = 20  # frames skipped before the step means (buffer pools filling, encoder lookahead)

DECODE = {"cpu": "avdec_h264", "cpu1": "avdec_h264 max-threads=1", "vpu": "vpudec frame-drop=false"}
SCALE = {"cpu": "videoscale ! video/x-raw,width=448,height=256",
         "g2d": "imxvideoconvert_g2d ! video/x-raw,width=448,height=256"}
ENCODE = {"cpu": "x264enc bitrate=2000", "vpu": "vpuenc_h264 bitrate=2000"}
RELAY_CONVERSIONS = [  # relay.py INPUT_CONVERTERS["g2d"] and make_output_converter()
	"videoconvert ! video/x-raw,format=GRAY8,colorimetry=2:4:0:0",
	"videoconvert ! video/x-raw,format=RGBx ! videoconvert ! video/x-raw,format=I420",
]

# (decode, scale, encode, relay conversions); each runs without and then with FIFOs. None = step left out.
DECODE_ALONE = [("cpu", None, None, False), ("cpu1", None, None, False), ("vpu", None, None, False)]
PIPELINES = [
	("cpu", "cpu", "vpu", False),  # encode on the VPU: slides' steps 1-4
	("vpu", "cpu", "vpu", False),
	("vpu", "g2d", "vpu", False),
	("cpu", "g2d", "vpu", False),  # avdec check: decoding is the only CPU-heavy step
	("cpu1", "g2d", "vpu", False),
	("cpu", "cpu", "cpu", False),  # encode on the CPU: slides' step 5
	("vpu", "cpu", "cpu", False),
	("vpu", "g2d", "cpu", False),
	("vpu", "g2d", "vpu", True),  # the relay's real chain, with its colour conversions
]
CONFIGS = [c + (False,) for c in DECODE_ALONE] + [c + (q,) for c in PIPELINES for q in (False, True)]


def steps(dec, scale, enc, convert):
	# [(step name, GStreamer description)] in pipeline order, the decoder first.
	out = [(f"{dec} dec", DECODE[dec])]
	if scale:
		glue = scale == "g2d" and enc == "cpu" and not convert
		out.append((f"{scale} scale", SCALE[scale] + (" ! videoconvert ! video/x-raw,format=I420" if glue else "")))
	if convert:
		out += [("grey", RELAY_CONVERSIONS[0]), ("I420", RELAY_CONVERSIONS[1])]
	if enc:
		out.append((f"{enc} enc", ENCODE[enc]))
	return out


def label(dec, scale, enc, convert, queues):
	return (" -> fifo -> " if queues else " -> ").join(name for name, _ in steps(dec, scale, enc, convert))


def pipeline(video, dec, scale, enc, convert, queues):
	link = " ! queue ! " if queues else " ! "
	return (f"filesrc location={video} ! qtdemux ! h264parse ! "
	        + link.join(desc for _, desc in steps(dec, scale, enc, convert)) + " ! fakesink sync=false")


def cpu_busy():
	# Busy jiffies of all cores together (user + nice + system + irq + softirq), in seconds.
	v = [int(x) for x in open("/proc/stat").readline().split()[1:8]]
	return (v[0] + v[1] + v[2] + v[5] + v[6]) / 100


def run(desc):
	busy, start = cpu_busy(), time.monotonic()
	subprocess.run(["gst-launch-1.0", "-q", *desc.split()], check=True, stdout=subprocess.DEVNULL)
	return time.monotonic() - start, cpu_busy() - busy


def probed_run(video, config, frames):
	# One no-FIFO run with probes; returns {"probed_wall_ms": per frame, step name: mean ms per frame}.
	import gi
	gi.require_version("Gst", "1.0")
	from gi.repository import Gst
	Gst.init(None)
	pipe = Gst.parse_launch(pipeline(video, *config, False))
	it, elements = pipe.iterate_sorted(), []  # sink -> source, so each goes in at the front
	while True:
		res, element = it.next()
		if res != Gst.IteratorResult.OK:
			break
		elements.insert(0, element)
	factory = [e.get_factory().get_name() for e in elements]
	work = elements[factory.index(DECODE[config[0]].split()[0]) + 1:]
	work = [e for e in work if e.get_factory().get_name() not in ("capsfilter", "fakesink")]

	# Match the elements to the steps by their factory names.
	owner = []
	for name, desc in steps(*config)[1:]:
		owner += [name for part in desc.split("!") if not part.strip().startswith("video/")]
	assert len(owner) == len(work), ([e.get_factory().get_name() for e in work], owner)

	enter, times = {}, {e.get_name(): [] for e in work}

	def on_in(pad, info, name):
		enter[name] = time.perf_counter()
		return Gst.PadProbeReturn.OK

	def on_out(pad, info, name):
		start = enter.pop(name, None)  # None while an encoder fills its lookahead
		if start is not None:
			times[name].append((time.perf_counter() - start) * 1000)
		return Gst.PadProbeReturn.OK

	for e in work:
		e.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_in, e.get_name())
		e.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_out, e.get_name())

	start = time.monotonic()
	pipe.set_state(Gst.State.PLAYING)
	msg = pipe.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
	wall = time.monotonic() - start
	pipe.set_state(Gst.State.NULL)
	if msg.type == Gst.MessageType.ERROR:
		sys.exit(f"error: {msg.parse_error()[0].message}")

	result = {"probed_wall_ms": wall / frames * 1000}
	for e, step in zip(work, owner):
		result[step] = result.get(step, 0) + statistics.mean(times[e.get_name()][WARMUP:])
	return result


def main():
	parser = argparse.ArgumentParser(description="decode -> scale -> encode: whole pipelines, steps, predictions")
	parser.add_argument("--repeats", type=int, default=2, help="runs per configuration; the fastest counts")
	parser.add_argument("--res", default="1080p", choices=["480p", "720p", "1080p"])
	parser.add_argument("--frames", type=int, default=1251, help="frames in the video")
	parser.add_argument("--match", help="only configurations whose label contains this (decode alone always runs)")
	parser.add_argument("--probe", help=argparse.SUPPRESS)  # one probed run, in a child process
	args = parser.parse_args()
	video = VIDEOS / f"sintel_trailer-{args.res}.mp4"

	if args.probe:  # child: result as the last line
		print(json.dumps(probed_run(video, tuple(json.loads(args.probe)), args.frames)))
		return

	OUT.mkdir(parents=True, exist_ok=True)
	out = OUT / f"pipeline_bench_{args.res}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
	print(f"{args.res}, {args.frames} frames, fastest of {args.repeats}; ms per frame; "
	      f"predicted = sum of steps (no FIFOs) or slowest step (FIFOs)")
	print(f"{'configuration':72} {'ms':>6} {'pred':>6} {'CPU s':>6} {'cores':>5}  steps")
	decode_ms, step_ms = {}, {}  # decoder -> decode-alone ms; (dec, scale, enc, convert) -> {step: ms}
	with open(out, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["configuration", "decode", "scale", "encode", "convert", "fifos", "wall_s", "ms_per_frame",
		                 "fps", "cpu_s", "cores_busy", "all_runs_wall_s", "steps_ms", "prediction", "predicted_ms",
		                 "probed_wall_ms"])
		for config in CONFIGS:
			*base, queues = config
			base = tuple(base)
			name = label(*config)
			if args.match and base[1] and args.match not in name:
				continue
			runs = [run(pipeline(video, *config)) for _ in range(args.repeats)]
			wall, cpu = min(runs)
			ms = wall / args.frames * 1000

			probed = ""
			if not base[1]:  # decode alone: the decode step
				decode_ms[base[0]] = ms
			elif not queues:  # steps after the decoder, from a probed run of this pipeline
				child = subprocess.run([sys.executable, "-u", __file__, "--res", args.res, "--frames", str(args.frames),
				                        "--probe", json.dumps(base)], capture_output=True, text=True)
				if child.returncode != 0:
					sys.exit(f"probed run of {name} failed (exit {child.returncode}):\n{child.stderr[-2000:]}")
				result = json.loads(child.stdout.splitlines()[-1])
				probed = f"{result.pop('probed_wall_ms'):.2f}"
				step_ms[base] = {f"{base[0]} dec": decode_ms[base[0]], **result}

			parts = step_ms.get(base, {f"{base[0]} dec": ms})
			kind = "max" if queues else "sum"
			predicted = max(parts.values()) if queues else sum(parts.values())
			shown = "  ".join(f"{k} {v:.2f}" for k, v in parts.items())
			print(f"{name:72} {ms:6.2f} {predicted:6.2f} {cpu:6.2f} {cpu / wall:5.2f}  {kind}: {shown}", flush=True)
			writer.writerow([name, base[0], base[1] or "", base[2] or "", base[3], queues, f"{wall:.3f}", f"{ms:.2f}",
			                 f"{args.frames / wall:.1f}", f"{cpu:.2f}", f"{cpu / wall:.2f}",
			                 " ".join(f"{w:.2f}" for w, _ in runs), ";".join(f"{k}={v:.3f}" for k, v in parts.items()),
			                 kind, f"{predicted:.2f}", probed])
	print(f"saved {out}")


if __name__ == "__main__":
	main()
