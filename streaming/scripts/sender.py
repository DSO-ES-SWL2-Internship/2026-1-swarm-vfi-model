import argparse
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR.parent / "videos"

VIDEO_PATH = str(VIDEOS_DIR / "sintel_trailer-480p.mp4")

Gst.init(None)

# demuxer -> parser dynamic pad callback (same reason as sender.py's qtdemux:
# qtdemux's output pads only exist once it has actually parsed the container)
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


parser_args = argparse.ArgumentParser()
parser_args.add_argument("--host", default="0.0.0.0", help="address to listen on (0.0.0.0 = all interfaces)")
parser_args.add_argument("--port", type=int, default=5000)
args = parser_args.parse_args()

# TCP variant: swaps rtph264pay/udpsink for mpegtsmux/tcpserversink. RTP's
# sequencing/jitter-buffer machinery exists to cope with UDP's unordered,
# lossy delivery -- on a reliable TCP byte stream that's dead weight, so we
# drop RTP entirely and use MPEG-TS instead, which is designed to survive
# being carried over a plain byte stream (has its own internal framing/sync
# bytes so a demuxer can resync even if it joins mid-stream).
source = make_element("filesrc", "source")
demuxer = make_element("qtdemux", "demuxer")
parser = make_element("h264parse", "parser")
muxer = make_element("mpegtsmux", "muxer")
tcpserversink = make_element("tcpserversink", "tcpserversink")

pipeline = Gst.Pipeline.new("sender-tcp-pipeline")

elements = [source, demuxer, parser, muxer, tcpserversink]
if not pipeline or not all(elements):
	print("Failed to create pipeline or one of its elements")
	exit(-1)

for el in elements:
	pipeline.add(el)

if not source.link(demuxer):
	print("Elements from source to demuxer could not be linked")
	exit(-1)

if not link_many(parser, muxer, tcpserversink):
	print("Elements from parser to tcpserversink could not be linked")
	exit(-1)

source.set_property("location", VIDEO_PATH)
tcpserversink.set_property("host", args.host)
tcpserversink.set_property("port", args.port)
tcpserversink.set_property("sync", True)  # pace to the buffers' own timestamps, same reasoning as sender.py

demuxer.connect("pad-added", on_pad_added, parser)

print(f"Listening on {args.host}:{args.port} -- waiting for receiver to connect...")

if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
	print("Unable to set the pipeline to the playing state.")
	exit(-1)

bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
if msg is not None:
	if msg.type == Gst.MessageType.ERROR:
		err, debug_info = msg.parse_error()
		print(f"error from element {msg.src.get_name()}: {err.message}")
	else:
		print("End-Of-Stream reached.")

pipeline.set_state(Gst.State.NULL)
print("Sender finished.")
