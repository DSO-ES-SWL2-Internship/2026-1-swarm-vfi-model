import argparse
import time
from pathlib import Path

# GStreamer
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

# Math/ML
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent.parent.parent / "artifacts"  # machine-learning/artifacts -- three levels up

MODEL_PATH = str(MODELS_DIR / "toy_unet_int8.onnx")

Gst.init(None)

# tsdemux's video src pad only exists once it has parsed the stream header --
# same reason sender.py needs this for its demuxer.
def on_pad_added(src, new_pad, parser):
	sink_pad = parser.get_static_pad("sink")

	caps = new_pad.get_current_caps()
	structure = caps.get_structure(0)
	if structure.get_name().startswith("video/"):
		if new_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
			print("Failed to link demuxer pad to parser")


class Passthrough:
	def __init__(self, appsrc):
		self.appsrc = appsrc

	def on_new_preroll(self, sink):
		sample = sink.pull_preroll()
		if not sample:
			return Gst.FlowReturn.ERROR
		self.appsrc.push_buffer(sample.get_buffer())
		return Gst.FlowReturn.OK

	def on_new_sample(self, sink):
		sample = sink.pull_sample()
		if not sample:
			return Gst.FlowReturn.ERROR
		self.appsrc.push_buffer(sample.get_buffer())
		return Gst.FlowReturn.OK

	def on_eos(self, sink):
		self.appsrc.end_of_stream()


class FrameHold:
	def __init__(self, appsrc):
		self.appsrc = appsrc
		self.prev_buffer = None

	def hold(self, prev_buf, curr_buf):
		success, map_info = prev_buf.map(Gst.MapFlags.READ)
		data = map_info.data
		prev_buf.unmap(map_info)

		held_buf = Gst.Buffer.new_wrapped(data)
		held_buf.pts = (prev_buf.pts + curr_buf.pts) // 2
		return held_buf

	def on_new_preroll(self, sink):
		sample = sink.pull_preroll()
		if not sample:
			return Gst.FlowReturn.ERROR
		buffer = sample.get_buffer()
		self.appsrc.push_buffer(buffer)
		self.prev_buffer = buffer
		return Gst.FlowReturn.OK

	def on_new_sample(self, sink):
		sample = sink.pull_sample()
		if not sample:
			return Gst.FlowReturn.ERROR
		buffer = sample.get_buffer()

		held = self.hold(self.prev_buffer, buffer)
		self.appsrc.push_buffer(held)
		self.appsrc.push_buffer(buffer)

		self.prev_buffer = buffer
		return Gst.FlowReturn.OK

	def on_eos(self, sink):
		self.appsrc.end_of_stream()


# ONNX runtime implementation of toy unet model -- this is the actual point of the relay node:
# raw pixels in, VFI-interpolated raw pixels out. NPU acceleration happens here (or doesn't,
# depending on which onnxruntime build/execution providers are installed on the device this
# runs on) -- see standalone_inference.py for isolating and verifying NPU usage separately.
class ToyUnetBlenderONNX:

	def __init__(self, appsrc):
		self.appsrc = appsrc
		self.prev_buffer = None

		self.session = ort.InferenceSession(MODEL_PATH)
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self.session.get_outputs()[0].name

	def infer(self, buf1, buf2):
		success1, map1 = buf1.map(Gst.MapFlags.READ)
		success2, map2 = buf2.map(Gst.MapFlags.READ)

		frame1 = np.frombuffer(map1.data, dtype=np.uint8).reshape(256, 448)
		frame2 = np.frombuffer(map2.data, dtype=np.uint8).reshape(256, 448)

		buf1.unmap(map1)
		buf2.unmap(map2)

		# QDQ-quantized ONNX model takes float32 in [0, 1] at its external boundary --
		# Quantize/Dequantize ops are inserted internally around each op instead, so no manual
		# zero-point arithmetic is needed here, just the same normalization training used.
		f1 = frame1.astype(np.float32)[..., None] / 255.0
		f2 = frame2.astype(np.float32)[..., None] / 255.0
		input_tensor = np.concatenate([f1, f2], axis=-1)[None, ...]

		output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
		output_pixels = (output[0, ..., 0] * 255).clip(0, 255).astype(np.uint8)

		return output_pixels

	def blend(self, buf1, buf2):
		predicted_pixels = self.infer(buf1, buf2)
		predicted_buf = Gst.Buffer.new_wrapped(predicted_pixels.tobytes())
		predicted_buf.pts = (buf1.pts + buf2.pts) // 2
		return predicted_buf

	def on_new_preroll(self, sink):
		sample = sink.pull_preroll()
		if not sample:
			return Gst.FlowReturn.ERROR
		buffer = sample.get_buffer()
		self.appsrc.push_buffer(buffer)
		self.prev_buffer = buffer
		return Gst.FlowReturn.OK

	def on_new_sample(self, sink):
		sample = sink.pull_sample()
		if not sample:
			return Gst.FlowReturn.ERROR
		buffer = sample.get_buffer()

		predicted = self.blend(self.prev_buffer, buffer)
		self.appsrc.push_buffer(predicted)
		self.appsrc.push_buffer(buffer)

		self.prev_buffer = buffer
		return Gst.FlowReturn.OK

	def on_eos(self, sink):
		self.appsrc.end_of_stream()


class LinearBlender:
	def __init__(self, appsrc):
		self.appsrc = appsrc
		self.prev_buffer = None

	def blend(self, buf1, buf2):
		success1, map1 = buf1.map(Gst.MapFlags.READ)
		success2, map2 = buf2.map(Gst.MapFlags.READ)

		frame1 = np.frombuffer(map1.data, dtype=np.uint8)
		frame2 = np.frombuffer(map2.data, dtype=np.uint8)

		blended_pixels = ((frame1.astype(np.uint16) + frame2.astype(np.uint16)) // 2).astype(np.uint8)

		buf1.unmap(map1)
		buf2.unmap(map2)

		blended_buf = Gst.Buffer.new_wrapped(blended_pixels.tobytes())
		blended_buf.pts = (buf1.pts + buf2.pts) // 2
		return blended_buf

	def on_new_preroll(self, sink):
		sample = sink.pull_preroll()
		if not sample:
			return Gst.FlowReturn.ERROR

		buffer = sample.get_buffer()
		self.appsrc.push_buffer(buffer)
		self.prev_buffer = buffer
		return Gst.FlowReturn.OK

	def on_new_sample(self, sink):
		sample = sink.pull_sample()
		if not sample:
			return Gst.FlowReturn.ERROR

		buffer = sample.get_buffer()

		blended = self.blend(self.prev_buffer, buffer)
		self.appsrc.push_buffer(blended)
		self.appsrc.push_buffer(buffer)

		self.prev_buffer = buffer
		return Gst.FlowReturn.OK

	def on_eos(self, sink):
		self.appsrc.end_of_stream()


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


def wait_for_eos_or_error(pipeline, label):
	bus = pipeline.get_bus()
	msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
	got_eos = False

	if msg is not None:
		if msg.type == Gst.MessageType.ERROR:
			err, debug_info = msg.parse_error()
			print(f"{label}: error from element {msg.src.get_name()}: {err.message}")
			print(f"{label}: debugging information: {debug_info or 'none'}")
		elif msg.type == Gst.MessageType.EOS:
			print(f"{label}: End-Of-Stream reached.")
			got_eos = True
		else:
			print(f"{label}: unexpected message received.")

	return got_eos


TECHNIQUES = {
	"passthrough": Passthrough,
	"frame_hold": FrameHold,
	"linear_blend": LinearBlender,
	"toy_unet_onnx": ToyUnetBlenderONNX,
}

parser_args = argparse.ArgumentParser(description="iMX8 relay node: receives from the Pi, runs VFI, "
                                                    "re-encodes, and sends the result onward to the VM.")
parser_args.add_argument("technique", choices=TECHNIQUES.keys())
parser_args.add_argument("--host", type=str, required=True, help="Pi's IP address to connect to")
parser_args.add_argument("--port", type=int, default=5000, help="Pi's port")
parser_args.add_argument("--idle-timeout", type=int, default=10,
                          help="seconds with no frames arriving before assuming the Pi is done -- "
                               "TCP does deliver a real EOS when the sender closes the connection, but this "
                               "heuristic is kept as a fallback for an unclean disconnect.")
parser_args.add_argument("--listen-host", type=str, default="0.0.0.0",
                          help="address to listen on for the VM to connect (0.0.0.0 = all interfaces)")
parser_args.add_argument("--listen-port", type=int, default=5001,
                          help="port to listen on for the VM to connect -- defaults to 5001, "
                               "distinct from --port (5000), so this can be loopback-tested "
                               "against itself without a port clash")
args = parser_args.parse_args()

# Arm A: receive (from Pi) -> demux (TS) -> decode -> downscale -> VFI technique -> appsink.
# Identical to two_device/receiver.py's arm A -- same reasoning applies (see that file's comments
# for the dynamic-pad-linking pattern).
tcpclientsrc = make_element("tcpclientsrc", "tcpclientsrc")
demuxer = make_element("tsdemux", "demuxer")
parser = make_element("h264parse", "parser")
decoder = make_element("avdec_h264", "decoder")
convert = make_element("videoconvert", "convert")
scale = make_element("videoscale", "scale")
capsfilter = make_element("capsfilter", "capsfilter")
appsink = make_element("appsink", "appsink")

# Arm B: appsrc -> queue -> convert -> encoder -> muxer -> tcpserversink. Unlike
# two_device/receiver.py's arm B, this sends the VFI'd result onward to the VM over the network
# instead of writing it to a local file -- mpegtsmux (not mp4mux), matching sender.py's own
# choice of a streaming-friendly container over a file-container format.
#
# The queue is still needed for the same reason as two_device/receiver.py: appsrc.push_buffer()
# runs downstream synchronously in whatever thread calls it (arm A's own appsink callback
# thread), and queue decouples that from the thread actually encoding/sending downstream.
appsrc = make_element("appsrc", "appsrc")
queue = make_element("queue", "queue")
convert2 = make_element("videoconvert", "convert2")
# GRAY8 -> I420 happens in convert2 above (x264enc doesn't take raw GRAY8 directly).
encoder = make_element("x264enc", "encoder")
muxer = make_element("mpegtsmux", "muxer")
tcpserversink = make_element("tcpserversink", "tcpserversink")

pipeline = Gst.Pipeline.new("relay-pipeline")
pipeline2 = Gst.Pipeline.new("relay-pipeline-2")

elements = [tcpclientsrc, demuxer, parser, decoder, convert, scale, capsfilter, appsink,
            appsrc, queue, convert2, encoder, muxer, tcpserversink]
if not pipeline or not pipeline2 or not all(elements):
	print("Failed to create pipeline or one of its elements")
	exit(-1)

for el in [tcpclientsrc, demuxer, parser, decoder, convert, scale, capsfilter, appsink]:
	pipeline.add(el)
for el in [appsrc, queue, convert2, encoder, muxer, tcpserversink]:
	pipeline2.add(el)

# demuxer's src pad is dynamic (only appears once tsdemux has parsed the TS stream header), so
# it can't be linked up front -- link tcpclientsrc->demuxer now, demuxer->parser happens later
# via the pad-added callback, same pattern as sender.py's qtdemux.
if not tcpclientsrc.link(demuxer):
	print("tcpclientsrc could not be linked to demuxer")
	exit(-1)

if not link_many(parser, decoder, convert, scale, capsfilter, appsink):
	print("Elements from parser to appsink could not be linked")
	exit(-1)

if not link_many(appsrc, queue, convert2, encoder, muxer, tcpserversink):
	print("Elements from appsrc to tcpserversink could not be linked")
	exit(-1)

demuxer.connect("pad-added", on_pad_added, parser)

tcpclientsrc.set_property("host", args.host)
tcpclientsrc.set_property("port", args.port)
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)
appsrc.set_property("format", Gst.Format.TIME)
tcpserversink.set_property("host", args.listen_host)
tcpserversink.set_property("port", args.listen_port)
tcpserversink.set_property("sync", True)  # pace to the buffers' own timestamps, same reasoning as sender.py

caps = Gst.Caps.from_string("video/x-raw, width=448, height=256, format=GRAY8")
capsfilter.set_property("caps", caps)
appsrc.set_property("caps", caps)

## PERFORMANCE TIMING
# Track when the last frame arrived, so the main loop can detect "no frames for N seconds"
# and infer the Pi is done -- an inactivity heuristic instead of a fixed window, adapting to
# however long the stream actually is.
last_frame_time = time.time()

# First/last relay timestamps, captured via a pad probe on tcpserversink's sink pad -- runs
# directly in the pipeline's own buffer-flow, not a Python signal callback, so it's the closest
# we can get to "when did this frame actually get sent onward" without added Python-level
# scheduling latency. The gap between them is the throughput measure for this hop specifically.
first_relay_time = None
last_relay_time = None

# Plain progress counters (no logging, just visibility) -- arrived tracks receiving from the
# Pi, relayed tracks post-VFI output sent onward to the VM, which is the stage most likely to
# lag given VFI compute happens here.
arrived_count = 0
relayed_count = 0


def track_activity(handler):
	def wrapped(sink):
		global last_frame_time, arrived_count
		last_frame_time = time.time()
		arrived_count += 1
		if arrived_count == 1 or arrived_count % 120 == 0:
			print(f"[arrived: {arrived_count}]")
		return handler(sink)
	return wrapped


def on_tcpserversink_buffer(pad, info):
	global first_relay_time, last_relay_time, relayed_count
	now = time.time()
	if first_relay_time is None:
		first_relay_time = now
	last_relay_time = now
	relayed_count += 1
	if relayed_count == 1 or relayed_count % 120 == 0:
		print(f"[relayed: {relayed_count}]")
	return Gst.PadProbeReturn.OK


# TCP gives a real EOS when the Pi closes its end of the connection cleanly (tcpclientsrc sees
# the socket close and pushes EOS downstream) -- unlike UDP, which has no "sender is done"
# signal at all. That EOS reaches appsink and fires this handler, which finalizes arm B
# (pipeline2) via technique.on_eos -> appsrc.end_of_stream(). Track that it happened so the
# main loop below doesn't call end_of_stream() a second time.
stream_ended = False


def on_appsink_eos(sink):
	global stream_ended
	stream_ended = True
	technique.on_eos(sink)


technique = TECHNIQUES[args.technique](appsrc)
appsink.connect("new-sample", track_activity(technique.on_new_sample))
appsink.connect("new-preroll", track_activity(technique.on_new_preroll))
appsink.connect("eos", on_appsink_eos)

tcpserversink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_tcpserversink_buffer)

print(f"Connecting to Pi at {args.host}:{args.port}, technique={args.technique}...")
print(f"Listening on {args.listen_host}:{args.listen_port} for the VM to connect...")

if pipeline2.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
	print("Unable to set pipeline2 to the playing state.")
	exit(-1)
if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
	print("Unable to set the pipeline to the playing state.")
	exit(-1)

# Primary termination path: a real EOS on pipeline's bus, posted once tcpclientsrc sees the Pi
# close its end of the connection and that EOS has propagated all the way to appsink.
# idle_timeout is a fallback for the case TCP can't cover -- an unclean disconnect with no FIN
# ever sent, where neither EOS nor ERROR would otherwise arrive and the loop would hang forever.
bus = pipeline.get_bus()
while True:
	msg = bus.timed_pop_filtered(1 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
	if msg is not None:
		if msg.type == Gst.MessageType.ERROR:
			err, debug_info = msg.parse_error()
			print(f"pipeline: error from element {msg.src.get_name()}: {err.message}")
		else:
			print("pipeline: End-Of-Stream reached -- Pi closed the connection.")
		break
	if time.time() - last_frame_time > args.idle_timeout:
		print(f"No frames for {args.idle_timeout}s -- assuming Pi is gone (unclean disconnect), finalizing.")
		break

if not stream_ended:
	appsrc.end_of_stream()
wait_for_eos_or_error(pipeline2, "pipeline2")

pipeline.set_state(Gst.State.NULL)
pipeline2.set_state(Gst.State.NULL)

if first_relay_time is not None:
	print(f"Time from first to last frame relayed: {last_relay_time - first_relay_time:.3f}s")
else:
	print("No frames were relayed.")
print("Relay finished.")
