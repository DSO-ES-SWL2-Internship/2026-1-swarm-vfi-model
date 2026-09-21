import argparse
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR.parent.parent / "videos"  # streaming/videos -- two levels up

Gst.init(None)

# This is the VM's role in the 3-device topology -- unlike two_device/receiver.py, it does no
# VFI and never touches raw pixels at all: it just receives the iMX8's already-interpolated,
# already-encoded stream and remuxes it from TS (network-transport container) into MP4 (a
# seekable file container) for later playback/comparison. No decode, no appsrc/appsink Python
# bridge, no encoder -- a single linear GStreamer pipeline is all this needs.
def on_pad_added(src, new_pad, parser):
	sink_pad = parser.get_static_pad("sink")

	caps = new_pad.get_current_caps()
	structure = caps.get_structure(0)
	if structure.get_name().startswith("video/"):
		if new_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
			print("Failed to link demuxer pad to parser")


def make_element(factory, name):
	element = Gst.ElementFactory.make(factory, name)
	if not element:
		print(f"Failed to create element: {factory}")
	return element


def link_many(*elements):
	for src, dst in zip(elements, elements[1:]):
		if not src.link(dst):
			return False
	return True


parser_args = argparse.ArgumentParser(description="VM receiver for the 3-device topology: pure "
                                                    "remux to file, no VFI (the iMX8 already did that).")
parser_args.add_argument("--host", type=str, required=True, help="iMX8's IP address to connect to")
parser_args.add_argument("--port", type=int, default=5001, help="iMX8's listen port")
parser_args.add_argument("--idle-timeout", type=int, default=10,
                          help="seconds with no frames arriving before assuming the iMX8 is done -- "
                               "TCP does deliver a real EOS when the sender closes the connection, but this "
                               "heuristic is kept as a fallback for an unclean disconnect.")
parser_args.add_argument("--output", type=str, default=None,
                          help="path to save the received video (default: streaming/videos/received_3hop.mp4)")
args = parser_args.parse_args()
output_path = args.output or str(VIDEOS_DIR / "received_3hop.mp4")

tcpclientsrc = make_element("tcpclientsrc", "tcpclientsrc")
demuxer = make_element("tsdemux", "demuxer")
parser = make_element("h264parse", "parser")
muxer = make_element("mp4mux", "muxer")
filesink = make_element("filesink", "filesink")

pipeline = Gst.Pipeline.new("receiver-remux-pipeline")

elements = [tcpclientsrc, demuxer, parser, muxer, filesink]
if not pipeline or not all(elements):
	print("Failed to create pipeline or one of its elements")
	exit(-1)

for el in elements:
	pipeline.add(el)

# demuxer's src pad is dynamic (only appears once tsdemux has parsed the TS stream header), so
# it can't be linked up front -- same pattern as the other scripts' demuxers.
if not tcpclientsrc.link(demuxer):
	print("tcpclientsrc could not be linked to demuxer")
	exit(-1)

if not link_many(parser, muxer, filesink):
	print("Elements from parser to filesink could not be linked")
	exit(-1)

demuxer.connect("pad-added", on_pad_added, parser)

tcpclientsrc.set_property("host", args.host)
tcpclientsrc.set_property("port", args.port)
# sync=False -- no live viewer, and pacing to real-time pts would make the first-frame/
# last-frame throughput measurement below meaningless, same reasoning as two_device/receiver.py.
filesink.set_property("sync", False)
filesink.set_property("location", output_path)

## PERFORMANCE TIMING
last_frame_time = time.time()
first_write_time = None
last_write_time = None
arrived_count = 0
written_count = 0


def on_parser_buffer(pad, info):
	global last_frame_time, arrived_count
	last_frame_time = time.time()
	arrived_count += 1
	if arrived_count == 1 or arrived_count % 120 == 0:
		print(f"[arrived: {arrived_count}]")
	return Gst.PadProbeReturn.OK


def on_filesink_buffer(pad, info):
	global first_write_time, last_write_time, written_count
	now = time.time()
	if first_write_time is None:
		first_write_time = now
	last_write_time = now
	written_count += 1
	if written_count == 1 or written_count % 120 == 0:
		print(f"[written: {written_count}]")
	return Gst.PadProbeReturn.OK


# Pad probes run directly in the pipeline's own buffer-flow -- the closest we can get to "when
# did this buffer actually arrive/get written" without added Python-level scheduling latency.
# No appsrc/appsink bridge exists here to hang callbacks off, unlike the other two scripts.
parser.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_parser_buffer)
filesink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_filesink_buffer)

print(f"Connecting to iMX8 at {args.host}:{args.port}...")

if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
	print("Unable to set the pipeline to the playing state.")
	exit(-1)

# A single pipeline here (unlike the other two scripts' two-pipeline appsrc/appsink bridge) --
# there's no separate compute stage to drain after EOS, filesink is directly downstream in the
# same pipeline, so one EOS on this bus means everything (including the file) is finalized.
bus = pipeline.get_bus()
while True:
	msg = bus.timed_pop_filtered(1 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
	if msg is not None:
		if msg.type == Gst.MessageType.ERROR:
			err, debug_info = msg.parse_error()
			print(f"pipeline: error from element {msg.src.get_name()}: {err.message}")
		else:
			print("pipeline: End-Of-Stream reached -- iMX8 closed the connection.")
		break
	if time.time() - last_frame_time > args.idle_timeout:
		print(f"No frames for {args.idle_timeout}s -- assuming iMX8 is gone (unclean disconnect), finalizing output.")
		break

pipeline.set_state(Gst.State.NULL)

if first_write_time is not None:
	print(f"Time from first to last frame written: {last_write_time - first_write_time:.3f}s")
else:
	print("No frames were written.")
print("Receiver finished.")
