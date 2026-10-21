# Why is the CPU scaler slower on VPU frames than on ordinary memory, when both are cacheable and read
# equally fast in a straight copy (trace_buffers.py)? Suspect: the TLB. A 1080p frame spans ~900 4 KB pages,
# more than the A53's TLB holds, and a scaler jumping between rows touches many pages per output row;
# ordinary memory may instead sit in 2 MB huge pages (transparent huge pages), needing only a couple of entries.
#
# Run on the iMX8: python3 scaler_tlb.py [--decoder vpu|cpu] [480p|720p|1080p]
# (each case runs in its own process; one leaving GStreamer state behind crashed the next on the board)
#
# The same decoded frames are scaled twice, in two separate runs that differ only in the memory:
#   as-is   the decoder's own buffer goes to videoscale (VPU: a linux,cma dmabuf)
#   copy    a deep copy of it, in ordinary memory, goes to videoscale
# decode -> appsink | (frame or copy) -> appsrc -> videoscale -> 448x256 -> fakesink
# The copy is made outside the timed part, so only videoscale's own work is compared.
#
# Per frame, between the buffer entering videoscale (sink pad) and leaving it (src pad), on that thread only:
#   ms         wall time. videoscale works on one thread, so unlike subtracting pipeline times
#              (scale_cost.sh) no other thread's work overlaps it.
#   cycles     CPU cycles, as a check on the time.
#   L1D miss   level 1 data cache refills: the access-pattern factor (cache lines fetched).
#   dTLB miss  data TLB refills: the page-translation factor. Each one needs a page-table walk.
# The counters come from the CPU's performance monitoring unit via perf_event_open, the same counters
# `perf stat` reads. They count user and kernel work, so mapping the dmabuf (cache maintenance) is included.
# --decoder cpu is the control: its frames are ordinary memory, so as-is and copy should match.
#
# Needs python3-gi with the GstVideo typelib. If the counters can't be opened, only ms is reported.

import argparse
import ctypes
import json
import os
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstApp, GstVideo  # noqa: F401 (GstApp gives appsrc/appsink their methods)

VIDEOS = Path(__file__).resolve().parent.parent.parent / "videos"
DECODERS = {"vpu": "vpudec frame-drop=false", "cpu": "avdec_h264"}
WARMUP = 20  # frames skipped before the medians (first maps, buffer pools filling)

# perf_event_open (linux/perf_event.h). Hardware cache events: config = cache | (op << 8) | (result << 16).
SYS_PERF_EVENT_OPEN = {"aarch64": 241, "x86_64": 298}[os.uname().machine]
PERF_TYPE_HARDWARE, PERF_TYPE_HW_CACHE = 0, 3
EVENTS = {  # name: (type, config)
	"cycles": (PERF_TYPE_HARDWARE, 0),  # PERF_COUNT_HW_CPU_CYCLES
	"L1D miss": (PERF_TYPE_HW_CACHE, 0 | (0 << 8) | (1 << 16)),  # L1D, read, miss
	"dTLB miss": (PERF_TYPE_HW_CACHE, 3 | (0 << 8) | (1 << 16)),  # DTLB, read, miss
}
libc = ctypes.CDLL(None, use_errno=True)


def open_counters():
	# One counter per event on the calling thread (pid 0, any CPU), counting from now. {} if not allowed.
	counters = {}
	for name, (kind, config) in EVENTS.items():
		attr = struct.pack("IIQ", kind, 112, config).ljust(112, b"\0")  # perf_event_attr, all flags 0
		fd = libc.syscall(SYS_PERF_EVENT_OPEN, attr, 0, -1, -1, 0)
		if fd < 0:
			print(f"  ({name} counter unavailable: {os.strerror(ctypes.get_errno())})")
			continue
		counters[name] = fd
	return counters


def read_counters(counters):
	return {name: struct.unpack("Q", os.read(fd, 8))[0] for name, fd in counters.items()}


def ordinary_copy(buf):
	# The frame copied into ordinary (malloc'd) memory, with its timestamps and layout (video meta).
	new = Gst.Buffer.new_wrapped(buf.extract_dup(0, buf.get_size()))
	new.pts, new.dts, new.duration = buf.pts, buf.dts, buf.duration
	meta = GstVideo.buffer_get_video_meta(buf)
	if meta:
		GstVideo.buffer_add_video_meta_full(new, meta.flags, meta.format, meta.width, meta.height,
		                                    meta.n_planes, list(meta.offset), list(meta.stride))
	return new


def run(source, copy):
	# Scales every frame once; returns a list of per-frame {"ms": ..., counter: ...}.
	dec = Gst.parse_launch(f"{source} ! appsink name=out emit-signals=true sync=false max-buffers=2")
	scale = Gst.parse_launch(
		"appsrc name=in format=time block=true max-buffers=2 ! videoscale name=scale ! "
		"video/x-raw,width=448,height=256 ! fakesink sync=false")
	appsrc = scale.get_by_name("in")
	first = [True]

	def on_sample(sink):
		sample = sink.emit("pull-sample")
		buf = sample.get_buffer()
		if first[0]:
			appsrc.set_caps(sample.get_caps())
		if copy:
			buf = ordinary_copy(buf)
		if first[0]:
			mem = buf.peek_memory(0)
			print(f"  videoscale input: {mem.allocator.mem_type if mem.allocator else '?'}, "
			      f"{sample.get_caps().to_string().split(',', 1)[1][:70]}...")
			first[0] = False
		appsrc.emit("push-buffer", buf)
		return Gst.FlowReturn.OK

	dec.get_by_name("out").connect("new-sample", on_sample)
	dec.get_by_name("out").connect("eos", lambda sink: appsrc.emit("end-of-stream"))

	counters, frames, start = {}, [], {}

	def on_in(pad, info):
		if not counters and "tried" not in start:  # first call: open the counters on this streaming thread
			start["tried"] = True
			counters.update(open_counters())
		start.update(t=time.perf_counter(), c=read_counters(counters), thread=threading.get_ident())
		return Gst.PadProbeReturn.OK

	def on_out(pad, info):
		end, c = time.perf_counter(), read_counters(counters)
		assert threading.get_ident() == start["thread"], "videoscale pushed from another thread"
		frames.append({"ms": (end - start["t"]) * 1000, **{k: c[k] - start["c"][k] for k in c}})
		return Gst.PadProbeReturn.OK

	element = scale.get_by_name("scale")
	element.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_in)
	element.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_out)

	loop = GLib.MainLoop()
	for pipe in (dec, scale):
		bus = pipe.get_bus()
		bus.add_signal_watch()
		bus.connect("message::error", lambda b, m: (print("  error:", m.parse_error()[0].message), loop.quit()))
	scale.get_bus().connect("message::eos", lambda b, m: loop.quit())
	scale.set_state(Gst.State.PLAYING)
	dec.set_state(Gst.State.PLAYING)
	loop.run()
	for pipe in (dec, scale):  # watches left on the main loop crash the next run (SIGSEGV on the board)
		pipe.set_state(Gst.State.NULL)
		pipe.get_bus().remove_signal_watch()
	for fd in counters.values():
		os.close(fd)
	return frames


def huge_pages():
	try:
		thp = Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text().strip()
	except OSError:
		thp = "not available"
	rollup = Path("/proc/self/smaps_rollup").read_text()
	anon = next((l.split(":")[1].strip() for l in rollup.splitlines() if l.startswith("AnonHugePages")), "?")
	return thp, anon


def main():
	parser = argparse.ArgumentParser(description="Per-frame cost and counters of videoscale on decoder memory vs a copy")
	parser.add_argument("--decoder", choices=DECODERS, default="vpu")
	parser.add_argument("res", nargs="?", default="1080p", choices=["480p", "720p", "1080p"])
	parser.add_argument("--case", choices=["as-is", "copy"], help=argparse.SUPPRESS)  # one case, in a child process
	args = parser.parse_args()

	if args.case:  # child: run one case, print its medians as the last line
		Gst.init(None)
		video = VIDEOS / f"sintel_trailer-{args.res}.mp4"
		source = f"filesrc location={video} ! qtdemux ! h264parse ! {DECODERS[args.decoder]}"
		frames = run(source, args.case == "copy")[WARMUP:]
		print(f"  {len(frames)} frames timed; huge pages in use by this process now: {huge_pages()[1]}")
		print(json.dumps({k: statistics.median(f[k] for f in frames) for k in frames[0]} if frames else {}))
		return

	# Each case runs in its own process, so nothing one run leaves behind can affect the other.
	print(f"{args.decoder} decode, {args.res}; transparent huge pages: {huge_pages()[0]}")
	results = {}
	for label in ("as-is", "copy"):
		print(f"{label}:", flush=True)
		child = subprocess.run([sys.executable, "-u", __file__, "--decoder", args.decoder, args.res, "--case", label],
		                       capture_output=True, text=True)
		lines = [l for l in child.stdout.splitlines() if l.startswith("  ")]
		print("\n".join(lines))
		if child.returncode != 0:
			sys.exit(f"  {label} run failed (exit {child.returncode}):\n{child.stderr[-2000:]}")
		results[label] = json.loads(child.stdout.splitlines()[-1])

	keys = list(results["as-is"])
	print(f"\nmedian per frame inside videoscale ({args.decoder} decode, {args.res}):")
	print(f"  {'':8}" + "".join(f"{k:>14}" for k in keys))
	for label, med in results.items():
		print(f"  {label:8}" + "".join(f"{med[k]:14.2f}" if k == "ms" else f"{med[k]:14.0f}" for k in keys))
	a, c = results["as-is"], results["copy"]
	print(f"  {'ratio':8}" + "".join(f"{a[k] / c[k]:14.2f}" if c[k] else f"{'-':>14}" for k in keys))


if __name__ == "__main__":
	main()
