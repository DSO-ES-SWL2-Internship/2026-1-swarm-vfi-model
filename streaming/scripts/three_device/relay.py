"""iMX8 relay node of the 3-device setup: receives the Pi's video, runs VFI on every pair of frames, re-encodes the
result and serves it to the VM.

Two GStreamer pipelines, joined by the VFI technique in Python:
  Arm A  source -> demux -> parse -> decode -> queue -> scale -> grey -> queue -> appsink
  Arm B  appsrc -> queue -> I420 -> encode -> mux -> tcpserversink (to the VM)
The appsink callback hands each frame to the technique, which pushes the original and the interpolated frames into
appsrc. Queues give the step after them their own thread, so decoding, scaling/converting, the VFI and encoding work
on different frames at once instead of taking turns in one thread (pipeline_bench.py: with queues a pipeline runs at
the pace of its slowest step, not the sum of its steps).

Usage (on the iMX8; run_relay.sh fills in the defaults):
  python3 relay.py toy_unet_tflite --host <Pi's IP>
  python3 relay.py toy_unet_tflite --input sintel_trailer-480p-12fps.mp4 --output fake   # local file, flat out
The original full-range relay (rgb/lut range modes, the ONNX technique) is at git tag before-simplify.
"""
from __future__ import annotations  # hints are not evaluated, so they can name classes defined further down

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

# GStreamer
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp

# Math/ML
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR.parent.parent / "videos"  # streaming/videos
MODELS_DIR = SCRIPT_DIR.parent.parent.parent / "artifacts"  # machine-learning/artifacts -- three levels up

# The U-Net trained on video-range grey (scripts/training.py --range video): the decoder's luma plane is video range
# (16-235), so frames go to the model and back to the encoder without any range conversion. Same accuracy as the
# full-range model (scripts/compare_range.py).
TFLITE_MODEL_PATH = str(MODELS_DIR / "toy_unet_int8_video.tflite")
VX_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"  # board-only -- NXP's eIQ NPU delegate for TIM-VX
CPU_THREADS = 4  # TFLite threads when the model runs on the CPU (--inference cpu): all four A53 cores
# The NPU driver can save its compiled graph (*.nb) here and reuse it on later runs, skipping the
# ~15s first-invoke compile (NXP i.MX ML User's Guide, sec. 8.1.3). Keyed by a hash of the model,
# so a re-exported model recompiles automatically.
VX_CACHE_DIR = MODELS_DIR / "vx_cache"
WIDTH, HEIGHT = 448, 256  # the model's frame size

Gst.init(None)


# ---------------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="iMX8 relay node: receives from the Pi, runs VFI, "
	                                             "re-encodes, and sends the result onward to the VM.")
	# VFI
	parser.add_argument("technique", choices=TECHNIQUES.keys())

	# Network communication
	parser.add_argument("--host", type=str, help="Pi's IP address to connect to (not needed with --input)")
	parser.add_argument("--port", type=int, default=5000, help="Pi's port")
	parser.add_argument("--idle-timeout", type=int, default=10,
	                    help="seconds with no frames arriving before assuming the Pi is done -- "
	                         "TCP does deliver a real EOS when the sender closes the connection, but this "
	                         "heuristic is kept as a fallback for an unclean disconnect.")
	parser.add_argument("--listen-host", type=str, default="0.0.0.0",
	                    help="address to listen on for the VM to connect (0.0.0.0 = all interfaces)")
	parser.add_argument("--listen-port", type=int, default=5001,
	                    help="port to listen on for the VM to connect -- defaults to 5001, "
	                         "distinct from --port (5000), so this can be loopback-tested "
	                         "against itself without a port clash")
	
	# Hardware offloading/configuration
	parser.add_argument("--decoder", choices=["auto", "cpu", "vpu"], default="auto",
	                    help="H.264 decoder: auto = vpu (hardware) if installed, else cpu (avdec_h264)")
	parser.add_argument("--scaler", choices=["auto", "cpu", "ocl", "g2d"], default="auto",
	                    help="scale to the model's 448x256 on: cpu (videoscale, the default), ocl (GPU) or g2d "
	                         "(see SCALERS)")
	parser.add_argument("--encoder", choices=["auto", "vpu", "x264", "mpeg4"], default="auto",
	                    help="re-encode codec: auto = vpu (hardware) if installed, else x264, else mpeg4")
	parser.add_argument("--inference", choices=["npu", "cpu"], default="npu",
	                    help="where toy_unet_tflite runs the model: npu (vx_delegate, if installed) or cpu, e.g. "
	                         "for the all-CPU comparison in the hardware acceleration demo")
	parser.add_argument("--convert-threads", type=int, default=1,
	                    help="threads per videoconvert (n-threads); 1 = GStreamer's default")
	
	# For measuring the relay on its own (streaming/scripts/test/relay_bench.py): a local file in, flat out.
	parser.add_argument("--input", type=str,
	                    help="read an MP4 file (path, or a name in streaming/videos) instead of the Pi's stream")
	parser.add_argument("--output", choices=["vm", "fake"], default="vm",
	                    help="vm = serve the result to the VM; fake = discard it after encoding (fakesink, not "
	                         "paced, no waiting for the VM)")
	parser.add_argument("--no-queues", action="store_true",
	                    help="leave out the queues between the steps, for comparison")
	
	args = parser.parse_args()
	if not args.input and not args.host:
		parser.error("--host is needed unless --input is given")
	return args


# ---------------------------------------------------------------------------------------------------------------------
# VFI techniques: each gets the decoded grey frames from appsink and pushes its output frames into appsrc
# ---------------------------------------------------------------------------------------------------------------------
class Passthrough:
	FPS_MULTIPLIER = 1  # forwards frames 1:1; every other technique doubles them

	def __init__(self, appsrc: GstApp.AppSrc | TimedAppsrc) -> None:
		self.appsrc = appsrc

	def on_new_sample(self, sink: GstApp.AppSink | VideoRangeSink) -> Gst.FlowReturn:
		sample = sink.pull_sample()
		if not sample:
			return Gst.FlowReturn.ERROR
		self.appsrc.push_buffer(sample.get_buffer())
		return Gst.FlowReturn.OK

	def on_eos(self, sink: GstApp.AppSink) -> None:
		self.appsrc.end_of_stream()


class Doubler(Passthrough):
	# Doubles the frame rate: for each new frame, pushes a frame between it and the previous one (middle()), then the
	# new frame itself. The subclasses only say how the middle frame is made.
	FPS_MULTIPLIER = 2

	def __init__(self, appsrc: GstApp.AppSrc | TimedAppsrc) -> None:
		super().__init__(appsrc)
		self.prev_buffer: Gst.Buffer | None = None

	def middle(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	def on_new_sample(self, sink: GstApp.AppSink | VideoRangeSink) -> Gst.FlowReturn:
		sample = sink.pull_sample()
		if not sample:
			return Gst.FlowReturn.ERROR
		buffer = sample.get_buffer()
		if self.prev_buffer is not None:  # the first frame has nothing to interpolate from yet
			self.appsrc.push_buffer(self.middle_buffer(self.prev_buffer, buffer)) # push VFI-ed middle frame
		self.appsrc.push_buffer(buffer) # push new frame
		self.prev_buffer = buffer
		return Gst.FlowReturn.OK

	def middle_buffer(self, buf1: Gst.Buffer, buf2: Gst.Buffer) -> Gst.Buffer:
		frame1, frame2 = read_frame(buf1), read_frame(buf2)
		out = Gst.Buffer.new_wrapped(self.middle(frame1, frame2).tobytes())
		out.pts = (buf1.pts + buf2.pts) // 2
		return out


def read_frame(buf: Gst.Buffer) -> np.ndarray:
	# The grey frame in a GRAY8 buffer, as a 256x448 uint8 array (a copy, so the buffer can be unmapped).
	ok, info = buf.map(Gst.MapFlags.READ)
	frame = np.frombuffer(info.data, dtype=np.uint8).reshape(HEIGHT, WIDTH).copy()
	buf.unmap(info)
	return frame


class FrameHold(Doubler):
	def middle(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
		return frame1


class LinearBlender(Doubler):
	def middle(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
		return ((frame1.astype(np.uint16) + frame2) // 2).astype(np.uint8)

class ToyUnetBlenderTFLite(Doubler):
	def __init__(self, appsrc: GstApp.AppSrc | TimedAppsrc, inference: str = "npu") -> None:
		super().__init__(appsrc)
		self.total_infer_time = 0.0
		self.infer_calls = 0

		# tflite_runtime + vx_delegate exist only on the iMX8; elsewhere (e.g. the VM) fall back to
		# full TensorFlow's interpreter on CPU, so this technique still runs as a CPU baseline.
		try:
			from tflite_runtime.interpreter import Interpreter, load_delegate
		except ImportError:
			import tensorflow as tf
			Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

		# Request for VX delegate for running model on NPU (vs CPU)
		delegates = []
		if inference == "npu" and Path(VX_DELEGATE_PATH).exists():
			# Adds NPU delegate: supported layers run on the NPU
			VX_CACHE_DIR.mkdir(exist_ok=True)
			os.environ.setdefault("VIV_VX_ENABLE_CACHE_GRAPH_BINARY", "1")  # read by the driver at load
			os.environ.setdefault("VIV_VX_CACHE_BINARY_GRAPH_DIR", str(VX_CACHE_DIR))
			delegates.append(load_delegate(VX_DELEGATE_PATH))
			print("toy_unet_tflite: running on the NPU (vx_delegate)")
		elif inference == "cpu":
			# No delegate to be added: CPU as asked
			print(f"toy_unet_tflite: running on the CPU ({CPU_THREADS} threads), as asked (--inference cpu)")
		else:
			# NPU asked but no delegate here: falls back to CPU
			print(f"toy_unet_tflite: {VX_DELEGATE_PATH} not found -- running on CPU, NOT the NPU")
		print(f"toy_unet_tflite: model {Path(TFLITE_MODEL_PATH).name}")
		self.interpreter = Interpreter(model_path=TFLITE_MODEL_PATH, experimental_delegates=delegates,
		                               num_threads=None if delegates else CPU_THREADS)
		self.interpreter.allocate_tensors()
		self.input_detail = self.interpreter.get_input_details()[0]
		self.output_detail = self.interpreter.get_output_details()[0]
		# middle() turns uint8 pixels into the model's int8 input by a zero-point shift alone, which is only right
		# if the input scale is 1/255 (export_tflite.py calibrates the input range to 0-1 for that).
		in_scale = self.input_detail["quantization"][0]
		if abs(in_scale * 255 - 1) > 1e-4:
			sys.exit(f"toy_unet_tflite: input scale {in_scale} is not 1/255; re-export the model with export_tflite.py")

		# Invoke the model to cache the graph, prevent long stall before first frame
		print("toy_unet_tflite: warming up the model (~15s on the NPU the first time; cached after that)...")
		warmup_start = time.perf_counter()
		dummy_input = np.zeros(self.input_detail["shape"], dtype=self.input_detail["dtype"])
		self.interpreter.set_tensor(self.input_detail["index"], dummy_input)
		self.interpreter.invoke()
		print(f"toy_unet_tflite: warm-up took {time.perf_counter() - warmup_start:.1f}s")

	def middle(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
		# Fully INT8 in/out model -- both scales here are exactly 1/255 and 1/256, so the
		# quantized value for an already-uint8 pixel is just a zero-point shift, no real
		# scaling/rounding distortion. See export_tflite.py for how this model was produced.
		in_zero_point = self.input_detail["quantization"][1]
		f1_q = (frame1.astype(np.int16) + in_zero_point).astype(np.int8)
		f2_q = (frame2.astype(np.int16) + in_zero_point).astype(np.int8)
		self.interpreter.set_tensor(self.input_detail["index"], np.stack([f1_q, f2_q], axis=-1)[None, ...])

		infer_start = time.perf_counter()
		self.interpreter.invoke()
		self.total_infer_time += time.perf_counter() - infer_start
		self.infer_calls += 1

		output_q = self.interpreter.get_tensor(self.output_detail["index"])[0, ..., 0]
		out_zero_point = self.output_detail["quantization"][1]
		return (output_q.astype(np.int16) - out_zero_point).astype(np.uint8)


TECHNIQUES = {
	"passthrough": Passthrough,
	"frame_hold": FrameHold,
	"linear_blend": LinearBlender,
	"toy_unet_tflite": ToyUnetBlenderTFLite,  # NPU (vx_delegate) -- board-only, needs tflite_runtime
}


def make_technique(name: str, appsrc: GstApp.AppSrc | TimedAppsrc, inference: str) -> Passthrough:
	if name == "toy_unet_tflite":
		return ToyUnetBlenderTFLite(appsrc, inference)
	return TECHNIQUES[name](appsrc)


# Code for the G2D path
# The constants and the RangeLookup and VideoRangeSink are relevant when the G2D is used for scaling.
#
# The G2D outputs RGB only, so its grey is converted from RGB and comes out full range (0-255).
# This table compresses it to the video range (16-235) the model takes.
#
# The CPU and OCL scalers keep the decoder's YUV, whose luma plane is
# video range already, so they need no lookup.
FULL_TO_VIDEO = np.round(16 + np.arange(256) * 219 / 255).astype(np.uint8)


class RangeLookup:
	# Compresses a frame to video range by table lookup, and keeps the total time spent for the end-of-run report.
	total = 0.0
	frames = 0

	@classmethod
	def apply(cls, buf: Gst.Buffer, table: np.ndarray) -> Gst.Buffer:
		# New buffer with every pixel looked up in table; timestamps carried over.
		start = time.perf_counter()
		ok, info = buf.map(Gst.MapFlags.READ)
		pixels = table[np.frombuffer(info.data, dtype=np.uint8)]
		buf.unmap(info)
		out = Gst.Buffer.new_wrapped(pixels.tobytes())
		out.pts, out.dts, out.duration = buf.pts, buf.dts, buf.duration
		cls.total += time.perf_counter() - start
		cls.frames += 1
		return out


class VideoRangeSink:
	# Stands in for appsink on the G2D chain: the technique pulls samples as usual and gets frames already compressed
	# to video range. Techniques only call pull_sample().get_buffer(), so that is all this provides.
	class Sample:
		def __init__(self, buffer: Gst.Buffer) -> None:
			self.buffer = buffer

		def get_buffer(self) -> Gst.Buffer:
			return self.buffer

	def __init__(self, sink: GstApp.AppSink, table: np.ndarray = FULL_TO_VIDEO) -> None:
		self.sink, self.table = sink, table

	def pull_sample(self) -> VideoRangeSink.Sample | None:
		sample = self.sink.pull_sample()
		return self.Sample(RangeLookup.apply(sample.get_buffer(), self.table)) if sample else sample


# ---------------------------------------------------------------------------------------------------------------------
# Pipeline elements
# ---------------------------------------------------------------------------------------------------------------------
# Creates a GStreamer element
def make_element(factory: str, name: str) -> Gst.Element | None:
	element = Gst.ElementFactory.make(factory, name)
	if not element:
		print(f"Failed to create element: {factory}")
	return element


# auto prefers the iMX8's hardware decoder (vpudec, VPU), else avdec_h264 (CPU, gst-libav).
def make_decoder(choice: str) -> Gst.Element | None:
	if choice == "auto":
		choice = "vpu" if Gst.ElementFactory.find("vpudec") else "cpu"
	if choice == "vpu":
		decoder = make_element("vpudec", "decoder")
		if decoder:
			# Default true: adaptively drops frames when busy, which would silently skew measurements.
			decoder.set_property("frame-drop", False)
	else:
		decoder = make_element("avdec_h264", "decoder")
	print(f"Decoder: {choice}")
	return decoder


# Decode path, in two stages: the scaler changes the resolution to the model's 448x256 input, then the
# input converter changes the colour format to the GRAY8 the model takes. The encode path keeps the
# resolution, so it only has an output converter (make_output_converter).
#
# The model takes video-range grey (16-235), which is the luma (Y) plane of the decoder's YUV frames. So the scaler
# should keep YUV, and the input converter then only has to take the Y plane:
#   cpu  videoscale on the CPU (default). Keeps YUV. Measured cheapest overall at 480p (ocl_check.py): it costs
#        CPU time to scale, but nothing after it has to undo a format.
#   ocl  imxvideoconvert_ocl on the GPU (OpenCL): NV12 in, NV12 out. Little CPU, but slower than the G2D.
#   g2d  imxvideoconvert_g2d: the fastest scaler, but it outputs RGB only, so the CPU then converts RGB -> grey
#        (full range) and looks it up to video range (VideoRangeSink): more CPU in total than the cpu scaler.
#
# vpudec pads its frames to a multiple of 16 (480p: 854 -> 864 columns; 1080p: 1080 -> 1088 rows) and marks the real
# picture with a crop meta. The G2D and OCL honour it; videoscale doesn't and would scale the padding into the
# picture, so the cpu scaler crops it off first (videocrop, set from the first caps: see set_crop).
SCALERS = {
	"cpu": "videocrop name=crop ! videoscale ! capsfilter caps=video/x-raw,width=448,height=256",
	"ocl": "imxvideoconvert_ocl ! capsfilter caps=video/x-raw,format=NV12,width=448,height=256",
	"g2d": "imxvideoconvert_g2d ! capsfilter caps=video/x-raw,width=448,height=256",
}

# YUV -> GRAY8 takes the luma plane as it is (video range).

# RGB -> GRAY8 on the G2D chain: its colorimetry was chosen by measurement, as the setting whose grey matches 
# full-range training data; VideoRangeSink then compresses it to video range.
INPUT_CONVERTERS = {
	"cpu": "videoconvert ! capsfilter caps=video/x-raw,format=GRAY8",
	"ocl": "videoconvert ! capsfilter caps=video/x-raw,format=GRAY8",
	"g2d": "videoconvert ! capsfilter caps=\"video/x-raw,format=GRAY8,colorimetry=2:4:0:0\"",
}


def with_threads(description: str, threads: int) -> str:
	# videoconvert splits each frame across this many threads (n-threads; default 1).
	return description.replace("videoconvert", f"videoconvert n-threads={threads}") if threads > 1 else description


def make_bin(description: str, name: str) -> Gst.Bin | None:
	try:
		element = Gst.parse_bin_from_description(description, True)
	except GLib.Error as err:
		print(f"Failed to create {name}: {err.message}")
		return None
	element.set_name(name)
	return element


def make_scaler(choice: str) -> tuple[Gst.Bin | None, str]:
	if choice == "auto":
		choice = "cpu"
	print(f"Scaler: {choice}")
	return make_bin(SCALERS[choice], "scaler"), choice


def set_crop(crop: Gst.Element, stream_caps: Gst.Caps, decoded_caps: Gst.Caps) -> None:
	# videocrop's right/bottom = how much bigger the decoder's frame is than the stream's picture (its padding);
	# 0 for avdec_h264, which outputs the picture size.
	stream, decoded = stream_caps.get_structure(0), decoded_caps.get_structure(0)
	right = decoded.get_int("width")[1] - stream.get_int("width")[1]
	bottom = decoded.get_int("height")[1] - stream.get_int("height")[1]
	crop.set_property("right", max(right, 0))  # size of crop from the right edge
	crop.set_property("bottom", max(bottom, 0))  # size of crop from the bottom edge
	if right > 0 or bottom > 0:
		print(f"Decoder pads its frames: cropping {right} columns and {bottom} rows before scaling")


# Matched to the scaler: the format the scaler hands over decides what is left to convert.
def make_input_converter(scaler_choice: str, threads: int = 1) -> Gst.Bin | None:
	return make_bin(with_threads(INPUT_CONVERTERS[scaler_choice], threads), "input_converter")


# GRAY8 -> I420 for the encoder: the grey is video range, so this is a plain copy into the luma plane (chroma
# neutral). I420 is forced so encoders that also accept other formats (vpuenc_h264) never convert it themselves
# with their own, unverified range handling.
def make_output_converter(threads: int = 1) -> Gst.Bin | None:
	return make_bin(with_threads("videoconvert ! capsfilter caps=video/x-raw,format=I420", threads), "output_converter")


# auto prefers the iMX8's hardware encoder (vpuenc_h264, VPU), then x264enc (CPU), then avenc_mpeg4 (CPU, always
# available via gst-libav). All target ~2 Mbit/s so comparisons between them are fair.
def make_encoder(choice: str) -> Gst.Element | None:
	if choice == "auto":
		if Gst.ElementFactory.find("vpuenc_h264"):
			choice = "vpu"
		elif Gst.ElementFactory.find("x264enc"):
			choice = "x264"
		else:
			choice = "mpeg4"
	if choice == "vpu":
		encoder = make_element("vpuenc_h264", "encoder")
		if encoder:
			encoder.set_property("bitrate", 2000)  # kbps; default 0 = automatic
	elif choice == "x264":
		encoder = make_element("x264enc", "encoder")  # default bitrate 2048 kbps
	else:
		encoder = make_element("avenc_mpeg4", "encoder")
		if encoder:
			encoder.set_property("bitrate", 2_000_000)  # bits/s; default 200 kbit/s is visibly blocky
	print(f"Encoder: {choice}")
	return encoder


def link_many(*elements: Gst.Element) -> bool:
	for src, dst in zip(elements, elements[1:]):
		if not src.link(dst):
			return False
	return True


# ---------------------------------------------------------------------------------------------------------------------
# Building the two pipelines
# ---------------------------------------------------------------------------------------------------------------------
def assemble(name: str, elements: list[Gst.Element | None]) -> Gst.Pipeline:
	# A pipeline of these elements, linked in order; None entries (queues left out by --no-queues) are skipped.
	elements = [el for el in elements if el is not None]
	pipeline = Gst.Pipeline.new(name)
	for el in elements:
		pipeline.add(el)
	if not link_many(*elements):
		sys.exit(f"{name}: elements could not be linked")
	return pipeline


def build_arm_a(args: argparse.Namespace) -> SimpleNamespace:
	# Arm A: source -> demux -> decode -> queue -> scale -> convert to grey -> queue -> appsink -> (VFI).
	# The source is the Pi's stream (tcpclientsrc, MPEG-TS) or, with --input, a local MP4 file.
	if args.input:
		source = make_element("filesrc", "source")
		demuxer = make_element("qtdemux", "demuxer")
		video = Path(args.input)
		source.set_property("location", str(video if video.exists() else VIDEOS_DIR / video))
	else:
		source = make_element("tcpclientsrc", "source")
		demuxer = make_element("tsdemux", "demuxer")
		source.set_property("host", args.host)
		source.set_property("port", args.port)
	parser = make_element("h264parse", "parser")
	decoder = make_decoder(args.decoder)
	scaler, scaler_choice = make_scaler(args.scaler)
	input_converter = make_input_converter(scaler_choice, args.convert_threads)
	appsink = make_element("appsink", "appsink")
	# Queues give the step after them their own thread (--no-queues leaves them out, for comparison).
	decode_queue, vfi_queue = (None, None) if args.no_queues else \
		(make_element("queue", "decode_queue"), make_element("queue", "vfi_queue"))
	if not all([source, demuxer, parser, decoder, scaler, input_converter, appsink]) or \
			not args.no_queues and not (decode_queue and vfi_queue):
		sys.exit("Failed to create one of arm A's elements")

	# The demuxer's src pad is dynamic (it only appears once the demuxer has parsed the stream header), so the
	# source -> demuxer link is made here and demuxer -> parser once the pad appears (on_pad_added).
	pipeline = assemble("relay-pipeline",
	                    [parser, decoder, decode_queue, scaler, input_converter, vfi_queue, appsink])
	pipeline.add(source)
	pipeline.add(demuxer)
	if not source.link(demuxer):
		sys.exit("source could not be linked to demuxer")
	demuxer.connect("pad-added", on_pad_added, parser)

	appsink.set_property("emit-signals", True)
	appsink.set_property("sync", False)

	# The cpu scaler's videocrop is set from the first caps that reach it: the stream's picture size (h264parse)
	# against the decoder's frame size, which includes vpudec's padding.
	if scaler_choice == "cpu":
		decoder.get_static_pad("src").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, on_decoder_caps,
		                                        (parser, scaler.get_by_name("crop")))
	return SimpleNamespace(pipeline=pipeline, scaler=scaler, scaler_choice=scaler_choice,
	                       input_converter=input_converter, appsink=appsink)


def on_pad_added(demuxer: Gst.Element, new_pad: Gst.Pad, parser: Gst.Element) -> None:
	# Links the demuxer's video pad to the parser once it appears (same reason sender.py needs this for its demuxer).
	if new_pad.get_current_caps().get_structure(0).get_name().startswith("video/"):
		if new_pad.link(parser.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
			print("Failed to link demuxer pad to parser")


def on_decoder_caps(pad: Gst.Pad, info: Gst.PadProbeInfo,
                    parser_and_crop: tuple[Gst.Element, Gst.Element]) -> Gst.PadProbeReturn:
	event = info.get_event()
	if event.type == Gst.EventType.CAPS:
		parser, crop = parser_and_crop
		stream_caps = parser.get_static_pad("src").get_current_caps()
		if stream_caps:
			set_crop(crop, stream_caps, event.parse_caps())
	return Gst.PadProbeReturn.OK


def build_arm_b(args: argparse.Namespace) -> SimpleNamespace:
	# Arm B: appsrc -> queue -> output converter -> encoder -> muxer -> tcpserversink.
	#
	# appsrc already hands frames to arm B on its own thread: even with --no-queues the VFI (arm A's appsink callback)
	# and the encoder run on different threads (relay_stats' shared_thread). The queue after it only adds buffering
	# between the two; it is kept so that every step boundary has one.
	appsrc = make_element("appsrc", "appsrc")
	encode_queue = None if args.no_queues else make_element("queue", "encode_queue")
	output_converter = make_output_converter(args.convert_threads)
	encoder = make_encoder(args.encoder)
	if args.output == "vm":
		sinks = [make_element("mpegtsmux", "muxer"), make_element("tcpserversink", "sink")]
	else:
		sinks = [make_element("fakesink", "sink")]
	if not all([appsrc, output_converter, encoder, *sinks]) or not args.no_queues and not encode_queue:
		sys.exit("Failed to create one of arm B's elements")

	if args.output == "vm":
		sinks[1].set_property("host", args.listen_host)
		sinks[1].set_property("port", args.listen_port)
		sinks[1].set_property("sync", True)  # pace to the buffers' own timestamps, same reasoning as sender.py
	else:
		sinks[0].set_property("sync", False)
	appsrc.set_property("format", Gst.Format.TIME)
	# Block the VFI when arm B falls behind (appsrc's own buffer is full), instead of buffering without limit: when the
	# relay runs flat out (--output fake) the encoder then sets the pace, as it would with a queue.
	appsrc.set_property("block", True)
	appsrc.set_property("caps", Gst.Caps.from_string(f"video/x-raw, width={WIDTH}, height={HEIGHT}, format=GRAY8"))

	pipeline = assemble("relay-pipeline-2", [appsrc, encode_queue, output_converter, encoder, *sinks])
	return SimpleNamespace(pipeline=pipeline, appsrc=appsrc, output_converter=output_converter, encoder=encoder,
	                       server=sinks[-1] if args.output == "vm" else None)


def connect_vfi(a: SimpleNamespace, b: SimpleNamespace, technique: Passthrough, stats: RelayStats,
                stream_ended: threading.Event) -> None:
	# Joins the arms: arm A's appsink feeds the technique, which pushes into arm B's appsrc.
	def on_new_sample(sink: GstApp.AppSink) -> Gst.FlowReturn:
		# The G2D's grey is full range (it outputs RGB): compressed to the model's video range on the way in.
		return technique.on_new_sample(VideoRangeSink(sink) if a.scaler_choice == "g2d" else sink)

	# TCP gives a real EOS when the Pi closes its end of the connection cleanly (tcpclientsrc sees
	# the socket close and pushes EOS downstream) -- unlike UDP, which has no "sender is done"
	# signal at all. That EOS reaches appsink and fires this handler, which finalizes arm B
	# via technique.on_eos -> appsrc.end_of_stream(). stream_ended records that it happened, so main()
	# doesn't call end_of_stream() a second time.
	def on_eos(sink: GstApp.AppSink) -> None:
		stream_ended.set()
		technique.on_eos(sink)

	# appsink delivers its first frame twice: once as the preroll sample, then again as the first
	# regular sample. Handling both pushed frame 0 twice -- shifting the output one frame behind the
	# source and starting the stream with duplicate timestamps (PTS 0, and a fake "interpolated"
	# frame between frame 0 and itself, also PTS 0), which vpuenc_h264 turns into buffers with no PTS
	# that mp4mux then rejects. Only the regular samples are used; the preroll copy is discarded.
	def discard_preroll(sink: GstApp.AppSink) -> Gst.FlowReturn:
		sink.pull_preroll()
		return Gst.FlowReturn.OK

	# Encoders need the real output framerate -- without one, avenc_mpeg4 assumes 25 fps and drops
	# frames that collide on that grid. VFI techniques emit two frames per input frame, Passthrough one.
	def on_appsink_caps(pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
		event = info.get_event()
		if event.type == Gst.EventType.CAPS:
			ok, fps_n, fps_d = event.parse_caps().get_structure(0).get_fraction("framerate")
			if ok and fps_n > 0:
				fps_n *= technique.FPS_MULTIPLIER
				b.appsrc.set_property("caps", Gst.Caps.from_string(
					f"video/x-raw, width={WIDTH}, height={HEIGHT}, format=GRAY8, framerate={fps_n}/{fps_d}"))
				print(f"Output framerate: {fps_n}/{fps_d}")
		return Gst.PadProbeReturn.OK

	a.appsink.connect("new-sample", stats.timed_callback(on_new_sample))
	a.appsink.connect("new-preroll", discard_preroll)
	a.appsink.connect("eos", on_eos)
	a.appsink.get_static_pad("sink").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, on_appsink_caps)
	b.encoder.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, stats.on_encoded_buffer)


# ---------------------------------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------------------------------
def report_start_failure(pipeline: Gst.Pipeline, label: str) -> None:
	# set_state() only returns FAILURE; the actual reason (e.g. "Could not connect to
	# 172.20.10.2:5000") is posted as an ERROR message on the pipeline's bus.
	msg = pipeline.get_bus().timed_pop_filtered(0, Gst.MessageType.ERROR)
	if msg is None:
		print(f"{label}: failed to start (no error message posted)")
		return
	err, debug_info = msg.parse_error()
	print(f"{label}: failed to start -- {err.message}")
	print(f"{label}: debugging information: {debug_info or 'none'}")


def set_state_or_exit(pipeline: Gst.Pipeline, state: Gst.State, label: str) -> None:
	if pipeline.set_state(state) == Gst.StateChangeReturn.FAILURE:
		report_start_failure(pipeline, label)
		sys.exit(-1)


def wait_for_vm(b: SimpleNamespace, args: argparse.Namespace) -> None:
	# Wait for the VM before touching the Pi at all: arm B (tcpserversink) streams in real time
	# the moment it's PLAYING, whether or not anyone is connected, so any frame relayed before the VM
	# joins would be lost. PAUSED already opens the listening socket; PLAYING starts the clock. Not
	# starting arm A (the Pi-facing side) until then also means no decode/VFI work is wasted on
	# frames nobody downstream can receive yet.
	vm_connected = threading.Event()
	b.server.connect("client-added", lambda sink, client: vm_connected.set())
	set_state_or_exit(b.pipeline, Gst.State.PAUSED, "pipeline2")
	print(f"Listening on {args.listen_host}:{args.listen_port} for the VM to connect...")
	bus = b.pipeline.get_bus()
	while not vm_connected.wait(timeout=0.5):
		msg = bus.pop_filtered(Gst.MessageType.ERROR)
		if msg is not None:
			err, debug_info = msg.parse_error()
			sys.exit(f"pipeline2: error from element {msg.src.get_name()}: {err.message}")
	print("VM connected -- starting relay.")


def wait_for_end(a: SimpleNamespace, args: argparse.Namespace, stats: RelayStats) -> None:
	# Primary termination path: a real EOS on arm A's bus, posted once tcpclientsrc sees the Pi
	# close its end of the connection (or the file ends) and that EOS has propagated all the way to appsink.
	# idle_timeout is a fallback for the case TCP can't cover -- an unclean disconnect with no FIN
	# ever sent, where neither EOS nor ERROR would otherwise arrive and the loop would hang forever.
	bus = a.pipeline.get_bus()
	while True:
		msg = bus.timed_pop_filtered(1 * Gst.SECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
		if msg is not None:
			if msg.type == Gst.MessageType.ERROR:
				err, debug_info = msg.parse_error()
				print(f"pipeline: error from element {msg.src.get_name()}: {err.message}")
			else:
				print("pipeline: End-Of-Stream reached -- " + ("end of file." if args.input else "Pi closed the connection."))
			return
		if time.time() - stats.last_frame_time > args.idle_timeout:
			print(f"No frames for {args.idle_timeout}s -- assuming Pi is gone (unclean disconnect), finalizing.")
			return


def wait_for_eos_or_error(pipeline: Gst.Pipeline, label: str) -> None:
	msg = pipeline.get_bus().timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
	if msg is None:
		print(f"{label}: unexpected message received.")
	elif msg.type == Gst.MessageType.ERROR:
		err, debug_info = msg.parse_error()
		print(f"{label}: error from element {msg.src.get_name()}: {err.message}")
		print(f"{label}: debugging information: {debug_info or 'none'}")
	else:
		print(f"{label}: End-Of-Stream reached.")


# ---------------------------------------------------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------------------------------------------------
def board_cpu() -> float:
	# Busy time of all cores together (user + nice + system + irq + softirq), in seconds.
	v = [int(x) for x in open("/proc/stat").readline().split()[1:8]]
	return (v[0] + v[1] + v[2] + v[5] + v[6]) / 100


# Step times, for predicting the relay's speed from its steps (relay_bench.py). Each is the time from a frame
# entering a step's first element to it leaving the last one, the same way pipeline_bench.py times its steps.
# Without queues one thread runs a frame through the steps, so nothing overlaps. For an encoder that holds frames
# back (x264enc's lookahead) it is the time since the latest frame went in, not its work per frame.
class StepTimer:
	def __init__(self, first: Gst.Element, last: Gst.Element) -> None:
		self.enter, self.total, self.frames = None, 0.0, 0
		first.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self.on_in)
		last.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self.on_out)

	def on_in(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
		self.enter = time.perf_counter()
		return Gst.PadProbeReturn.OK

	def on_out(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
		if self.enter is not None:
			self.total += time.perf_counter() - self.enter
			self.frames += 1
		return Gst.PadProbeReturn.OK

	def ms(self) -> float | None:
		return self.total / self.frames * 1000 if self.frames else None


class TimedAppsrc:
	# Stands in for appsrc and adds up the time spent pushing frames into it (which, when appsrc is full, is time
	# spent waiting for arm B), so it can be taken out of the VFI step's time.
	def __init__(self, appsrc: GstApp.AppSrc) -> None:
		self.appsrc, self.push_time = appsrc, 0.0

	def push_buffer(self, buf: Gst.Buffer) -> Gst.FlowReturn:
		start = time.perf_counter()
		result = self.appsrc.push_buffer(buf)
		self.push_time += time.perf_counter() - start
		return result

	def __getattr__(self, name: str):  # end_of_stream() and anything else go to the real appsrc
		return getattr(self.appsrc, name)


class RelayStats:
	# Progress and timing for the whole run.
	#
	# First/last relay moments come from a pad probe on the encoder's src pad (one buffer per encoded frame) -- not on
	# tcpserversink, since mpegtsmux pushes buffer *lists* of TS packets, which a BUFFER probe never sees. The gap between
	# them is the throughput measure for this hop. Process CPU time (all threads, user + system) and the whole board's
	# busy time (/proc/stat, which also counts driver work for the hardware units) are taken at the same moments, so
	# the CPU cost covers only the frames flowing -- not start-up, model loading or the NPU warm-up. The live stream is
	# paced by the Pi, so options that save CPU show up there rather than in the frame rate.
	PROGRESS_INTERVAL = 5  # progress is also printed at least this often (s), so a slow run still shows it is moving

	def __init__(self) -> None:
		# last_frame_time lets wait_for_end() detect "no frames for N seconds" and infer the Pi is done -- an
		# inactivity heuristic instead of a fixed window, adapting to however long the stream actually is.
		self.last_frame_time = time.time()
		# arrived tracks receiving from the Pi, relayed tracks post-VFI output sent onward to the VM, which is the
		# stage most likely to lag given VFI compute happens here.
		self.arrived, self.relayed = 0, 0
		self.first = self.last = None  # (wall, process CPU, board CPU)
		self.callback_time = 0.0
		self.threads = {"vfi": set(), "encode": set()}
		self.last_progress = {}

	def progress(self, label: str, count: int) -> None:
		now = time.time()
		if count == 1 or count % 120 == 0 or now - self.last_progress.get(label, now) >= self.PROGRESS_INTERVAL:
			print(f"[{label}: {count}]")
			self.last_progress[label] = now

	def timed_callback(self, handler: Callable[[GstApp.AppSink], Gst.FlowReturn]
	                   ) -> Callable[[GstApp.AppSink], Gst.FlowReturn]:
		# Wraps the appsink callback: counts arriving frames and adds up the callback's time (the VFI step).
		def wrapped(sink: GstApp.AppSink) -> Gst.FlowReturn:
			self.last_frame_time = time.time()
			self.arrived += 1
			self.progress("arrived", self.arrived)
			self.threads["vfi"].add(threading.get_native_id())
			start = time.perf_counter()
			result = handler(sink)
			self.callback_time += time.perf_counter() - start
			return result
		return wrapped

	def on_encoded_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
		self.last = (time.time(), time.process_time(), board_cpu())
		if self.first is None:
			self.first = self.last
		self.relayed += 1
		self.threads["encode"].add(threading.get_native_id())
		self.progress("relayed", self.relayed)
		return Gst.PadProbeReturn.OK

	def report(self, technique: Passthrough, appsrc: TimedAppsrc, scale_timer: StepTimer,
	           encode_timer: StepTimer) -> None:
		# Prints the summary, and one relay_stats line (JSON) for relay_bench.py to read. Step times are per input
		# frame (VFI) or per output frame (scale/convert, encode).
		stats = {"frames_in": self.arrived, "frames_out": self.relayed}
		if self.first is not None and self.relayed > 1:
			wall_s, cpu_s, board_s = (last - first for first, last in zip(self.first, self.last))
			print(f"Time from first to last frame relayed: {wall_s:.3f}s")
			# 100% = one core fully busy (the board has 4). Excludes work in kernel threads and drivers
			# outside this process (VPU/G2D/NPU hardware time is not CPU time at all).
			if wall_s > 0:
				print(f"Relay CPU time: {cpu_s:.2f}s = {cpu_s / wall_s * 100:.0f}% of one core, "
				      f"{cpu_s / (self.relayed - 1) * 1000:.2f} ms per relayed frame")
			stats.update(wall_s=wall_s, process_cpu_s=cpu_s, board_cpu_s=board_s)
		else:
			print("No frames were relayed.")

		# Only toy_unet_tflite has an inference step to report.
		if getattr(technique, "infer_calls", 0) > 0:
			stats["infer_ms"] = technique.total_infer_time / technique.infer_calls * 1000
			print(f"Mean inference latency ({technique.infer_calls} calls): {stats['infer_ms']:.2f} ms/frame")
		if RangeLookup.frames:
			stats["lookup_ms"] = RangeLookup.total / RangeLookup.frames * 1000
			print(f"Mean range lookup: {stats['lookup_ms']:.2f} ms/frame ({RangeLookup.frames} frames)")
		if self.arrived:
			stats["vfi_ms"] = (self.callback_time - appsrc.push_time) / self.arrived * 1000
		stats.update(scale_convert_ms=scale_timer.ms(), encode_ms=encode_timer.ms(),
		             threads={k: len(v) for k, v in self.threads.items()},
		             shared_thread=bool(self.threads["vfi"] & self.threads["encode"]))
		print("relay_stats: " + json.dumps(stats))


# ---------------------------------------------------------------------------------------------------------------------
def main() -> None:
	args = parse_args()
	print(f"Queues: {'no' if args.no_queues else 'yes'}, videoconvert threads: {args.convert_threads}")

	# Build relay pipelines
	a = build_arm_a(args)  # receive -> decode -> scale -> grey -> appsink
	b = build_arm_b(args)  # appsrc -> I420 -> encode -> VM (or fakesink)

	# Profiling
	stats = RelayStats()
	scale_timer = StepTimer(a.scaler, a.input_converter)
	encode_timer = StepTimer(b.output_converter, b.encoder)
	appsrc = TimedAppsrc(b.appsrc)

	# Final setup
	technique = make_technique(args.technique, appsrc, args.inference)
	stream_ended = threading.Event()
	connect_vfi(a, b, technique, stats, stream_ended)

	# Start the VM side first, then the Pi side.
	if args.output == "vm":
		wait_for_vm(b, args)
	set_state_or_exit(b.pipeline, Gst.State.PLAYING, "pipeline2")
	if args.input:
		print(f"Reading {args.input}, technique={args.technique}...")
	else:
		print(f"Connecting to Pi at {args.host}:{args.port}, technique={args.technique}...")
	set_state_or_exit(a.pipeline, Gst.State.PLAYING, "pipeline")

	wait_for_end(a, args, stats)
	if not stream_ended.is_set():
		b.appsrc.end_of_stream()
	wait_for_eos_or_error(b.pipeline, "pipeline2")
	a.pipeline.set_state(Gst.State.NULL)
	b.pipeline.set_state(Gst.State.NULL)

	stats.report(technique, appsrc, scale_timer, encode_timer)
	print("Relay finished.")


if __name__ == "__main__":
	main()
