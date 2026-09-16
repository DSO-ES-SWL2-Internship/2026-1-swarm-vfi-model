import argparse
import csv
import time

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp
import numpy as np
import onnxruntime as ort
import tensorflow as tf

Gst.init(None)


# tsdemux's video src pad only exists once it has parsed the stream header --
# same reason sender.py/sender_tcp.py need this for their demuxers.
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


class ToyUnetBlender:
	MODEL_PATH = "/home/dsointern/projects/intern1-swarm/machine-learning/artifacts/toy_unet_int8.tflite"

	def __init__(self, appsrc):
		self.appsrc = appsrc
		self.prev_buffer = None

		self.interpreter = tf.lite.Interpreter(model_path=self.MODEL_PATH)
		self.interpreter.allocate_tensors()
		self.input_detail = self.interpreter.get_input_details()[0]
		self.output_detail = self.interpreter.get_output_details()[0]

	def infer(self, buf1, buf2):
		success1, map1 = buf1.map(Gst.MapFlags.READ)
		success2, map2 = buf2.map(Gst.MapFlags.READ)

		frame1 = np.frombuffer(map1.data, dtype=np.uint8).reshape(256, 448)
		frame2 = np.frombuffer(map2.data, dtype=np.uint8).reshape(256, 448)

		buf1.unmap(map1)
		buf2.unmap(map2)

		in_zero_point = self.input_detail["quantization"][1]
		f1_q = (frame1.astype(np.int16) + in_zero_point).astype(np.int8)
		f2_q = (frame2.astype(np.int16) + in_zero_point).astype(np.int8)
		input_tensor = np.stack([f1_q, f2_q], axis=-1)[None, ...]

		self.interpreter.set_tensor(self.input_detail["index"], input_tensor)
		self.interpreter.invoke()
		output_q = self.interpreter.get_tensor(self.output_detail["index"])[0, ..., 0]

		out_zero_point = self.output_detail["quantization"][1]
		output_pixels = (output_q.astype(np.int16) - out_zero_point).astype(np.uint8)

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


class ToyUnetBlenderONNX:
	MODEL_PATH = "/home/dsointern/projects/intern1-swarm/machine-learning/artifacts/toy_unet_int8.onnx"

	def __init__(self, appsrc):
		self.appsrc = appsrc
		self.prev_buffer = None

		self.session = ort.InferenceSession(self.MODEL_PATH)
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self.session.get_outputs()[0].name

	def infer(self, buf1, buf2):
		success1, map1 = buf1.map(Gst.MapFlags.READ)
		success2, map2 = buf2.map(Gst.MapFlags.READ)

		frame1 = np.frombuffer(map1.data, dtype=np.uint8).reshape(256, 448)
		frame2 = np.frombuffer(map2.data, dtype=np.uint8).reshape(256, 448)

		buf1.unmap(map1)
		buf2.unmap(map2)

		# QDQ-quantized ONNX model takes float32 in [0, 1] at its external
		# boundary (unlike ToyUnetBlender's TFLite model, which is fully
		# INT8 in/out) -- Quantize/Dequantize ops are inserted internally
		# around each op instead, so no manual zero-point arithmetic is
		# needed here, just the same normalization training used.
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
	"toy_unet": ToyUnetBlender,
	"toy_unet_onnx": ToyUnetBlenderONNX,
}

parser_args = argparse.ArgumentParser()
parser_args.add_argument("technique", choices=TECHNIQUES.keys())
parser_args.add_argument("--host", type=str, required=True, help="sender (RPi) IP address to connect to")
parser_args.add_argument("--port", type=int, default=5000)
parser_args.add_argument("--idle-timeout", type=int, default=10,
                          help="seconds with no frames arriving before assuming the sender is done -- "
                               "TCP does deliver a real EOS when the sender closes the connection, but this "
                               "heuristic is kept as a fallback/consistency with the UDP receiver.")
args = parser_args.parse_args()

# Arm A: receive -> demux (TS) -> decode -> downscale -> VFI technique -> appsink
tcpclientsrc = make_element("tcpclientsrc", "tcpclientsrc")
demuxer = make_element("tsdemux", "demuxer")
parser = make_element("h264parse", "parser")
decoder = make_element("avdec_h264", "decoder")
convert = make_element("videoconvert", "convert")
scale = make_element("videoscale", "scale")
capsfilter = make_element("capsfilter", "capsfilter")
appsink = make_element("appsink", "appsink")

# Arm B: appsrc -> queue -> convert -> videosink (live playback window instead
# of writing a file -- no encode/mux needed, autovideosink can display raw
# frames directly, and sync=True paces display to each buffer's pts against
# the pipeline clock so it plays back at the right rate instead of flashing
# frames as fast as the VFI technique can produce them).
#
# The queue is not optional here, unlike with the old filesink output: appsrc.
# push_buffer() runs downstream synchronously, in whatever thread calls it --
# and that's the appsink "new-sample"/"new-preroll" callback, which fires on
# arm A's own GStreamer streaming thread, not the main Python thread. Without
# a queue, videosink's render() would get invoked from that foreign thread.
# filesink never cared which thread called it, but a windowing/GL sink like
# autovideosink does (GL contexts are thread-affine) -- calling into it from
# the wrong thread is exactly what caused the segfault. queue has its own
# internal thread, so it decouples the thread pushing buffers in (arm A) from
# the thread actually pulling and rendering them (a thread videosink owns).
appsrc = make_element("appsrc", "appsrc")
queue = make_element("queue", "queue")
convert2 = make_element("videoconvert", "convert2")
# ximagesink, not autovideosink -- autovideosink autoplugs to a GL-based sink
# (glimagesink) here, and that segfaults when sharing a process with an
# imported tensorflow (confirmed via isolated repro: identical two-pipeline
# appsink/appsrc bridge crashes with tensorflow+autovideosink, survives with
# tensorflow+ximagesink). tensorflow gets imported unconditionally at the top
# of this file even for techniques that don't use it, so any GL sink is
# unsafe here regardless of which --technique is selected.
videosink = make_element("ximagesink", "videosink")

pipeline = Gst.Pipeline.new("receiver-pipeline")
pipeline2 = Gst.Pipeline.new("receiver-pipeline-2")

elements = [tcpclientsrc, demuxer, parser, decoder, convert, scale, capsfilter, appsink,
            appsrc, queue, convert2, videosink]
if not pipeline or not pipeline2 or not all(elements):
	print("Failed to create pipeline or one of its elements")
	exit(-1)

for el in [tcpclientsrc, demuxer, parser, decoder, convert, scale, capsfilter, appsink]:
	pipeline.add(el)
for el in [appsrc, queue, convert2, videosink]:
	pipeline2.add(el)

# demuxer's src pad is dynamic (only appears once tsdemux has parsed the TS
# stream header), so it can't be linked up front the way the rest of the
# chain can -- link tcpclientsrc->demuxer now, demuxer->parser happens later
# via the pad-added callback, same pattern as sender.py/sender_tcp.py's qtdemux.
if not tcpclientsrc.link(demuxer):
	print("tcpclientsrc could not be linked to demuxer")
	exit(-1)

if not link_many(parser, decoder, convert, scale, capsfilter, appsink):
	print("Elements from parser to appsink could not be linked")
	exit(-1)

if not link_many(appsrc, queue, convert2, videosink):
	print("Elements from appsrc to videosink could not be linked")
	exit(-1)

demuxer.connect("pad-added", on_pad_added, parser)

tcpclientsrc.set_property("host", args.host)
tcpclientsrc.set_property("port", args.port)
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)
appsrc.set_property("format", Gst.Format.TIME)
videosink.set_property("sync", True)  # pace playback to each buffer's pts, not just display-as-fast-as-possible

caps = Gst.Caps.from_string("video/x-raw, width=448, height=256, format=GRAY8")
capsfilter.set_property("caps", caps)
appsrc.set_property("caps", caps)

# Track when the last frame arrived, so the main loop can detect "no frames
# for N seconds" and infer the sender is done -- an inactivity heuristic
# instead of a fixed window, adapting to however long the stream actually is.
last_frame_time = time.time()
frame_count = 0

# Stuttering-investigation instrumentation: two separate timestamp logs.
# "arrival" = when a decoded frame reaches appsink (arm A) -- reflects
# network/decode timing, upstream of any VFI processing. "render" = when a
# buffer actually reaches videosink's sink pad (arm B) -- reflects what the
# viewer actually sees, downstream of VFI and the queue. Kept separate on
# purpose: the network-isolation experiment cares about "arrival" intervals,
# the VFI-isolation experiment cares about "render" intervals, and comparing
# the two against each other is itself informative (e.g. VFI adding jitter
# that wasn't present in the arrival stream).
arrival_log = []
render_log = []


def track_activity(handler):
	def wrapped(sink):
		global last_frame_time, frame_count
		last_frame_time = time.time()
		frame_count += 1
		arrival_log.append((frame_count, last_frame_time))
		if frame_count == 1 or frame_count % 100 == 0:
			print(f"[{frame_count} frames received]")
		return handler(sink)
	return wrapped


render_frame_count = 0


def on_videosink_buffer(pad, info):
	global render_frame_count
	render_frame_count += 1
	render_log.append((render_frame_count, time.time()))
	return Gst.PadProbeReturn.OK


def write_frame_log():
	path = f"videos/frame_log_{args.technique}.csv"
	with open(path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["stage", "frame_index", "timestamp"])
		for index, ts in arrival_log:
			writer.writerow(["arrival", index, f"{ts:.6f}"])
		for index, ts in render_log:
			writer.writerow(["render", index, f"{ts:.6f}"])
	print(f"Wrote {len(arrival_log)} arrival + {len(render_log)} render timestamps to {path}")


# TCP gives a real EOS when the sender closes its end of the connection
# cleanly (tcpclientsrc sees the socket close and pushes EOS downstream) --
# unlike UDP, which has no "sender is done" signal at all. That EOS reaches
# appsink and fires this handler, which finalizes arm B (pipeline2) via
# technique.on_eos -> appsrc.end_of_stream(). Track that it happened so the
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

# Pad probes run directly in the pipeline's own buffer-flow, not in a Python
# signal callback -- this is the closest we can get to "when did this buffer
# actually reach the sink" without added Python-level scheduling latency.
videosink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_videosink_buffer)

print(f"Listening on port {args.port}, technique={args.technique}...")

if pipeline2.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
	print("Unable to set pipeline2 to the playing state.")
	exit(-1)
if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
	print("Unable to set the pipeline to the playing state.")
	exit(-1)

# Primary termination path: a real EOS on pipeline's bus, posted once tcpclientsrc
# sees the sender close its end of the connection and that EOS has propagated all
# the way to appsink. idle_timeout is a fallback for the case TCP can't cover --
# an unclean disconnect (crash, power loss, network drop) with no FIN ever sent,
# where neither EOS nor ERROR would otherwise arrive and the loop would hang forever.
bus = pipeline.get_bus()
while True:
	msg = bus.timed_pop_filtered(1 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
	if msg is not None:
		if msg.type == Gst.MessageType.ERROR:
			err, debug_info = msg.parse_error()
			print(f"pipeline: error from element {msg.src.get_name()}: {err.message}")
		else:
			print("pipeline: End-Of-Stream reached -- sender closed the connection.")
		break
	if time.time() - last_frame_time > args.idle_timeout:
		print(f"No frames for {args.idle_timeout}s -- assuming sender is gone (unclean disconnect), finalizing output.")
		break

if not stream_ended:
	appsrc.end_of_stream()
wait_for_eos_or_error(pipeline2, "pipeline2")

pipeline.set_state(Gst.State.NULL)
pipeline2.set_state(Gst.State.NULL)
write_frame_log()
print("Receiver finished.")
