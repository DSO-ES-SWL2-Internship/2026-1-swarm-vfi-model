# Shows how frames are handed from one hardware unit to the next, by looking at the buffers each GStreamer
# element passes on. Run on the iMX8: python3 trace_buffers.py [--decoder vpu|cpu] [--scaler g2d|cpu] [480p|720p|1080p]
#
# Pipeline (the relay's video path, without the model): decode -> scale to 448x256 -> VPU encode -> fakesink.
# For the first few frames at each hand-over it prints:
#   memory  what kind of memory holds the frame: "dmabuf" is a kernel buffer that hardware units share by
#           passing a file descriptor (no copy); "SystemMemory" is ordinary malloc'd memory, CPU only.
#   buffer  which dmabuf it is (its inode). The same number on both sides of a unit means it worked in
#           place; a dmabuf reaching the next unit at all means that unit read the frame where it lay (no copy).
#   heap    which kernel allocator made the dmabuf (from /proc/self/fdinfo). On the NXP BSP the
#           "...-uncached" heaps are mapped uncached for the CPU, so CPU reads of them go to DRAM every time.
#   phys    the frame's physical (DRAM) address, if the allocator reports it.
#   layout  the stride (bytes per row, including padding) and plane offsets, from the video meta.
# Then it times the CPU reading each frame (roadmap investigation 3, test 2), next to an ordinary buffer of
# the same size that the CPU has just written. If the decoder's frames read several times slower than the
# ordinary buffer, they are mapped uncached for the CPU: with no cache, every read waits on DRAM. Each frame is
# read twice: in cached memory the 2nd read can hit the cache (frames that fit in the A53's 512 KB L2, so
# the 448x256 ones); in uncached memory both reads are equally slow. The CPU decoder (--decoder cpu) is the
# control: its frames are ordinary memory, so they should read about as fast as the ordinary buffer.
#
# Needs python3-gi with the GstAllocators and GstVideo typelibs (gst-inspect-1.0 shows the elements).

import argparse
import os
import sys
import statistics
import time
from collections import defaultdict
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstAllocators", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstAllocators, GstVideo

VIDEOS = Path(__file__).resolve().parent.parent.parent / "videos"
DECODERS = {"vpu": "vpudec frame-drop=false", "cpu": "avdec_h264"}
SCALERS = {"g2d": "imxvideoconvert_g2d", "cpu": "videoscale"}
SHOW = 3  # frames printed per hand-over
TIMED = 100  # frames timed per hand-over


def describe(buf):
	# One line about where this buffer's frame lives.
	mem = buf.peek_memory(0)
	kind = mem.allocator.mem_type if mem.allocator else "?" # Get the allocator for the memory block (e.g. VpuMemory, dmabufheapsmem)
	ident, heap, phys = "-", "-", "-"

	# Report file descriptor properties for the dmabuf
	if GstAllocators.is_dmabuf_memory(mem):
		fd = GstAllocators.dmabuf_memory_get_fd(mem)
		ident = f"inode {os.fstat(fd).st_ino}"
		try:  # e.g. "exp_name:	linux,cma-uncached"
			fdinfo = Path(f"/proc/self/fdinfo/{fd}").read_text()
			heap = next((l.split(":", 1)[1].strip() for l in fdinfo.splitlines() if l.startswith("exp_name")), "-")
		except OSError:
			pass

	# Report physical memory address
	if GstAllocators.is_phys_memory(mem):
		phys = f"0x{GstAllocators.phys_memory_get_phys_addr(mem):x}"
	meta = GstVideo.buffer_get_video_meta(buf)

	# Video frame's layout information (format, dimensions, stride, plane offset)
	layout = f"{meta.format.value_nick} {meta.width}x{meta.height} stride {list(meta.stride[:meta.n_planes])} " \
	         f"offset {list(meta.offset[:meta.n_planes])}" if meta else "no video meta"
	return f"memory {kind:12} {ident:16} heap {heap:20} phys {phys:12} size {buf.get_size():8}  {layout}"


def time_reads(buf):
	# Maps the frame, then copies it into Python twice.
	# bytes() makes exactly one copy either way: info.data is
	# a view of the mapping where gst-python is installed (the board), but an already-copied bytes object
	# without it (the VM), and bytes() of a bytes object returns it as it is.
	#
	# Returns (map ms, first read MB/s, second read MB/s). Mapping a dmabuf includes the cache
	# maintenance the kernel does before a CPU read.
	start = time.perf_counter()
	ok, info = buf.map(Gst.MapFlags.READ)
	mapped = time.perf_counter()
	_ = bytes(info.data)
	first = time.perf_counter()
	_ = bytes(info.data)
	second = time.perf_counter()
	buf.unmap(info)
	size_mb = buf.get_size() / 1e6
	return (mapped - start) * 1000, size_mb / (first - mapped), size_mb / (second - first)


def ordinary_read(buf, ordinary, payload):
	# The same reads on an ordinary (malloc'd, cached) buffer of this frame's size, just rewritten by the CPU.
	if ordinary.get_size() != buf.get_size():
		ordinary.set_size(buf.get_size())
	ordinary.fill(0, payload[:buf.get_size()])
	return time_reads(ordinary)[1:]


def main():
	# Each copy makes a new Python bytes object. glibc gives allocations this large fresh pages from the kernel
	# every time, and faulting them in costs more than the copy itself, so restart with malloc reusing its heap.
	if "MALLOC_MMAP_THRESHOLD_" not in os.environ:
		env = dict(os.environ, MALLOC_MMAP_THRESHOLD_=str(128 << 20), MALLOC_TRIM_THRESHOLD_=str(256 << 20))
		os.execve(sys.executable, [sys.executable, *sys.argv], env)

	parser = argparse.ArgumentParser()
	parser.add_argument("--decoder", choices=DECODERS, default="vpu")
	parser.add_argument("--scaler", choices=SCALERS, default="g2d")
	parser.add_argument("res", nargs="?", choices=["480p", "720p", "1080p"], default="1080p")
	args = parser.parse_args()

	Gst.init(None)
	video = VIDEOS / f"sintel_trailer-{args.res}.mp4"
	pipeline = Gst.parse_launch(
		f"filesrc location={video} ! qtdemux ! h264parse ! {DECODERS[args.decoder]} name=dec ! "
		f"{SCALERS[args.scaler]} name=scale ! video/x-raw,width=448,height=256 ! vpuenc_h264 bitrate=2000 ! "
		f"fakesink")
	handovers = {"decoder -> scaler": pipeline.get_by_name("dec"), "scaler -> encoder": pipeline.get_by_name("scale")}

	shown, times = defaultdict(int), defaultdict(list)
	payload = bytes(range(256)) * ((8 << 20) // 256)  # 8 MB, more than a 1080p frame
	ordinary = Gst.Buffer.new_allocate(None, len(payload), None)

	# Instantiates a probe
	def probe(pad, info, label):
		buf = info.get_buffer()
		if shown[label] < SHOW:
			print(f"{label}, frame {shown[label]}: {describe(buf)}")
			shown[label] += 1
		if len(times[label]) < TIMED:
			times[label].append((*time_reads(buf), *ordinary_read(buf, ordinary, payload)))
		return Gst.PadProbeReturn.OK

	# Attaches the probe to a pad
	for label, element in handovers.items():
		element.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, label)

	print(f"{video.name}: {DECODERS[args.decoder]} ! {SCALERS[args.scaler]} ! vpuenc_h264\n")
	pipeline.set_state(Gst.State.PLAYING)
	bus = pipeline.get_bus()
	# Stop once every hand-over has its timed frames (or at the end of the video, or on an error).
	while min(len(times[label]) for label in handovers) < TIMED:
		msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
		if msg and msg.type == Gst.MessageType.ERROR:
			err, debug = msg.parse_error()
			print(f"Error: {err.message}\n{debug}")
			break
		if msg:
			break
	pipeline.set_state(Gst.State.NULL)

	print(f"\nCPU reads (map, then copy into Python twice), median over the frames, MB/s:")
	for label in handovers:
		if times[label]:
			map_ms, first, second, ord_first, ord_second = (statistics.median(column) for column in zip(*times[label]))
			print(f"  {label}: frame 1st {first:.0f}, 2nd {second:.0f} (map {map_ms:.2f} ms); "
			      f"ordinary buffer 1st {ord_first:.0f}, 2nd {ord_second:.0f} -> frame/ordinary {first / ord_first:.2f}")

if __name__ == "__main__":
	main()
