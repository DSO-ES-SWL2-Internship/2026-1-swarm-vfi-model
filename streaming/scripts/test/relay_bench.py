# What does each offload buy in the real relay, with the model? relay.py runs flat out from the 480p 12 fps trailer
# (--input, --output fake), moving one step at a time off the CPU, each stage without and with queues:
#   all CPU        avdec_h264, videoscale, the model on the CPU (TFLite, 4 threads), x264enc
#   + VPU decode   vpudec
#   + GPU scale    imxvideoconvert_ocl (--hw-scaler g2d for the G2D instead)
#   + NPU          the model on the NPU (vx_delegate)
#   + VPU encode   vpuenc_h264: the relay's default path, the baseline
#
# Run on the iMX8: python3 streaming/scripts/test/relay_bench.py [--repeats 2] [--hw-scaler ocl|g2d] [--match TEXT]
#
# Every run is its own relay.py process; its relay_stats line gives the wall time from the first to the last
# encoded frame (start-up, model loading and the NPU warm-up left out), the CPU time over the same span (the
# process, and the whole board from /proc/stat), and the step times. Times are per input frame (626): each input
# frame gives two output frames, so the encoder works twice per input frame.
#
# Steps and predictions:
#   decode        decode-alone time per frame (gst-launch, as in pipeline_bench.py): the decoder can't be timed inside
#                 the relay, since avdec_h264 decodes ahead on its own threads
#   scale+grey    scaler + input converter (relay_stats scale_convert_ms)
#   vfi           the appsink callback minus the time pushing into appsrc (vfi_ms): reading the frames, the model
#   encode x2     output converter + encoder (encode_ms), twice
#   Without queues one thread decodes, scales and runs the VFI, but appsrc pushes from its own thread, so the encoder
#   still runs alongside: predicted = max(decode + scale+grey + vfi, encode x2). With queues every step has its own
#   thread: predicted = the slowest step. Both use the step times of the no-queue run, where the CPU steps don't
#   compete for the cores (the queued run's step times are saved too).
#
# Where a prediction misses, that is the finding: avdec_h264 and x264enc work on threads of their own (x264enc's
# step time is how long the pipeline waits for it, not its work), and CPU steps that fill the 4 cores slow each
# other down.

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from pipeline_bench import DECODE, VIDEOS, run

RELAY = Path(__file__).resolve().parent.parent / "relay.py"
OUT = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "relay_bench"
VIDEO = "sintel_trailer-480p-12fps.mp4"
OUTPUT_PER_INPUT = 2  # toy_unet_tflite doubles the frame rate


def stages(hw_scaler):
	# [(stage name, relay.py options)], each stage adding one offload to the one before.
	options = {"decoder": "cpu", "scaler": "cpu", "inference": "cpu", "encoder": "x264"}
	out = [("all CPU", dict(options))]
	for name, key, value in [("+ VPU decode", "decoder", "vpu"), (f"+ {hw_scaler.upper()} scale", "scaler", hw_scaler),
	                         ("+ NPU", "inference", "npu"), ("+ VPU encode", "encoder", "vpu")]:
		options[key] = value
		out.append((name, dict(options)))
	return out


def run_relay(options, queues):
	# One relay.py run; returns its relay_stats.
	cmd = [sys.executable, "-u", str(RELAY), "toy_unet_tflite", "--input", VIDEO, "--output", "fake",
	       *(f"--{k}={v}" for k, v in options.items())] + ([] if queues else ["--no-queues"])
	child = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
	lines = [line for line in child.stdout.splitlines() if line.startswith("relay_stats: ")]
	if child.returncode != 0 or not lines:
		sys.exit(f"relay failed (exit {child.returncode}): {' '.join(cmd)}\n{child.stdout[-1500:]}\n"
		         f"{child.stderr[-1500:]}")
	return json.loads(lines[-1].split(" ", 1)[1])


def step_times(stats, decode_ms):
	# {step: ms per input frame}
	return {"decode": decode_ms, "scale+grey": stats["scale_convert_ms"], "vfi": stats["vfi_ms"],
	        "encode x2": OUTPUT_PER_INPUT * stats["encode_ms"]}


def predict(steps, queues):
	if queues:
		return max(steps.values())
	return max(steps["decode"] + steps["scale+grey"] + steps["vfi"], steps["encode x2"])


def main():
	parser = argparse.ArgumentParser(description="relay.py with the model, one offload at a time: speed, CPU, steps")
	parser.add_argument("--repeats", type=int, default=2, help="runs per configuration; the fastest counts")
	parser.add_argument("--hw-scaler", choices=["ocl", "g2d"], default="ocl", help="scaler of the '+ scale' stage on")
	parser.add_argument("--frames", type=int, default=626, help="frames in the video")
	parser.add_argument("--match", help="only stages whose name contains this")
	args = parser.parse_args()

	OUT.mkdir(parents=True, exist_ok=True)
	stem = OUT / f"relay_bench_{args.hw_scaler}_{time.strftime('%Y%m%d_%H%M%S')}"
	out_csv, out_txt = stem.with_suffix(".csv"), stem.with_suffix(".txt")
	report = []

	def say(line):
		print(line, flush=True)
		report.append(line)

	selected = [(name, options) for name, options in stages(args.hw_scaler) if not args.match or args.match in name]
	video = VIDEOS / VIDEO
	decode_ms = {}
	for dec in sorted({options["decoder"] for _, options in selected}):
		wall, _ = min(run(f"filesrc location={video} ! qtdemux ! h264parse ! {DECODE[dec]} ! fakesink sync=false")
		              for _ in range(args.repeats))
		decode_ms[dec] = wall / args.frames * 1000

	say(f"{VIDEO}, {args.frames} frames in, flat out, fastest of {args.repeats}; ms per input frame; predicted from "
	    f"the no-queue run's steps")
	say("decode alone: " + ", ".join(f"{DECODE[dec].split()[0]} {ms:.2f}" for dec, ms in decode_ms.items()))
	say(f"{'stage':16} {'queues':6} {'ms':>6} {'pred':>6} {'CPU ms':>7} {'cores':>5} {'infer':>6}  steps")
	with open(out_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["stage", "decoder", "scaler", "inference", "encoder", "queues", "wall_s", "ms_per_input",
		                 "predicted_ms", "process_cpu_s", "board_cpu_s", "cpu_ms_per_input", "cores_busy", "infer_ms",
		                 "lookup_ms", "steps_ms", "queued_steps_ms", "all_runs_wall_s"])
		for name, options in selected:
			steps = None
			for queues in (False, True):
				runs = [run_relay(options, queues) for _ in range(args.repeats)]
				stats = min(runs, key=lambda s: s["wall_s"])
				frames_in = (stats["frames_out"] - 1) / OUTPUT_PER_INPUT  # intervals between the first and last
				ms = stats["wall_s"] / frames_in * 1000
				measured = step_times(stats, decode_ms[options["decoder"]])
				if not queues:
					steps = measured
				predicted = predict(steps, queues)
				cpu_ms = stats["process_cpu_s"] / frames_in * 1000
				cores = stats["board_cpu_s"] / stats["wall_s"]
				shown = "  ".join(f"{k} {v:.2f}" for k, v in steps.items())
				say(f"{name:16} {'yes' if queues else 'no':6} {ms:6.2f} {predicted:6.2f} {cpu_ms:7.2f} {cores:5.2f} "
				    f"{stats.get('infer_ms', 0):6.2f}  {shown}")
				writer.writerow([name, options["decoder"], options["scaler"], options["inference"], options["encoder"],
				                 queues, f"{stats['wall_s']:.3f}", f"{ms:.2f}", f"{predicted:.2f}",
				                 f"{stats['process_cpu_s']:.2f}", f"{stats['board_cpu_s']:.2f}", f"{cpu_ms:.2f}",
				                 f"{cores:.2f}", f"{stats.get('infer_ms', 0):.2f}", f"{stats.get('lookup_ms', 0):.2f}",
				                 ";".join(f"{k}={v:.3f}" for k, v in steps.items()),
				                 ";".join(f"{k}={v:.3f}" for k, v in measured.items()),
				                 " ".join(f"{s['wall_s']:.2f}" for s in runs)])
				f.flush()
	out_txt.write_text("\n".join(report) + "\n")
	print(f"saved {out_csv} and {out_txt}")


if __name__ == "__main__":
	main()
