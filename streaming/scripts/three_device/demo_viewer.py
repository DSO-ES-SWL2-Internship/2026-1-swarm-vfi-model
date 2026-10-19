# Live demo window for the 3-device setup (runs on the VM, instead of receiver.py): the original video on the
# left, and on the right the stream the iMX8 sends back after the two hops (Pi -> iMX8 -> VM), side by side.
#
# The original is the VM's own copy of the Pi's video (sintel_trailer-480p.mp4), shown the way the model sees
# it -- scaled to 448x256 and in grey -- unless "Colour original" is ticked. It starts when the first processed
# frame arrives, so the two sides run roughly together (the processed side lags by the network and relay
# delay; it is not frame-exact). The processed side shows frames as they arrive: with an interpolating
# technique it has twice the frame rate, which the fps counters show.
#
# Usage: start the sender on the Pi and the relay on the iMX8 as usual (run_sender.sh, run_relay.sh), then
#   python3 demo_viewer.py [--host 172.20.10.10] [--port 5001]
# and press Connect (or pass --connect). The relay exits when the video ends, so restart it before pressing Connect again.

import argparse
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gst, Gtk

VIDEO = Path(__file__).resolve().parent.parent.parent / "videos" / "sintel_trailer-480p.mp4"
SIZE = "video/x-raw,width=448,height=256"
SCALE = 2  # each side is shown at 896x512


class Side:
	# One video panel: a GStreamer pipeline drawing into a gtksink widget, with a caption and an fps counter.
	def __init__(self, title):
		self.title = title
		self.pipeline = None
		self.frames, self.window_start, self.window_frames = 0, time.time(), 0
		self.caption = Gtk.Label(label=title)
		self.stats = Gtk.Label(label="-")
		self.slot = Gtk.Box()
		self.slot.set_size_request(448 * SCALE, 256 * SCALE)
		self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
		for widget in (self.caption, self.slot, self.stats):
			self.box.pack_start(widget, False, False, 0)

	def start(self, description, on_first_frame=None):
		self.stop()
		self.pipeline = Gst.parse_launch(description)
		sink = self.pipeline.get_by_name("sink")
		self.slot.foreach(self.slot.remove)
		widget = sink.get_property("widget")
		widget.set_size_request(448 * SCALE, 256 * SCALE)
		self.slot.pack_start(widget, True, True, 0)
		widget.show()
		self.frames, self.window_start, self.window_frames = 0, time.time(), 0
		self.on_first_frame = on_first_frame
		sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self.on_frame)
		bus = self.pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message::error", self.on_error)
		bus.connect("message::eos", lambda bus, msg: self.stats.set_text(f"{self.frames} frames -- finished"))
		self.pipeline.set_state(Gst.State.PLAYING)

	def stop(self):
		if self.pipeline:
			self.pipeline.get_bus().remove_signal_watch()
			self.pipeline.set_state(Gst.State.NULL)
			self.pipeline = None

	def on_frame(self, pad, info):
		# Runs in the streaming thread, so the first-frame callback is handed to the GTK thread.
		self.frames += 1
		self.window_frames += 1
		if self.frames == 1 and self.on_first_frame:
			GLib.idle_add(self.on_first_frame)
		return Gst.PadProbeReturn.OK

	def update_stats(self):
		now = time.time()
		if self.pipeline and now - self.window_start >= 1:
			fps = self.window_frames / (now - self.window_start)
			self.stats.set_text(f"{self.frames} frames, {fps:.1f} fps")
			self.window_start, self.window_frames = now, 0

	def on_error(self, bus, msg):
		err, _ = msg.parse_error()
		self.stats.set_text(f"Error: {err.message}")
		self.stop()


class DemoWindow(Gtk.Window):
	def __init__(self, host, port):
		super().__init__(title="VFI demo: original vs after two hops")
		self.connect("destroy", self.quit)
		self.original = Side("Original (VM's copy of the Pi's video)")
		self.processed = Side("Processed (Pi -> iMX8 -> VM)")

		# Add buttons to GUI
		self.host = Gtk.Entry(text=host, width_chars=15)
		self.port = Gtk.SpinButton.new_with_range(1, 65535, 1)
		self.port.set_value(port)
		self.colour = Gtk.CheckButton(label="Colour original")
		connect = Gtk.Button(label="Connect")
		connect.connect("clicked", self.on_connect)
		stop = Gtk.Button(label="Stop")
		stop.connect("clicked", lambda button: self.stop_all("Stopped."))
		self.status = Gtk.Label(label="Start the sender and relay, then press Connect.", xalign=0)

		# Add video stream to GUI
		controls = Gtk.Box(spacing=8)
		for widget in (Gtk.Label(label="iMX8"), self.host, Gtk.Label(label="port"), self.port, self.colour, connect, stop):
			controls.pack_start(widget, False, False, 0)
		videos = Gtk.Box(spacing=12)
		videos.pack_start(self.original.box, True, True, 0)
		videos.pack_start(self.processed.box, True, True, 0)
		layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
		for widget in (controls, videos, self.status):
			layout.pack_start(widget, False, False, 0)
		self.add(layout)
		GLib.timeout_add(500, self.tick)

	def on_connect(self, button):
		self.stop_all()
		host, port = self.host.get_text().strip(), int(self.port.get_value())
		self.status.set_text(f"Connecting to {host}:{port}, waiting for the first frame...")
		# decodebin picks the decoder for whichever codec the relay sends (H.264 from the VPU, or MPEG-4).
		self.processed.start(f"tcpclientsrc host={host} port={port} ! tsdemux ! decodebin ! videoconvert ! "
		                     f"gtksink name=sink sync=false", on_first_frame=self.start_original)

	def start_original(self):
		grey = "" if self.colour.get_active() else "videoconvert ! video/x-raw,format=GRAY8 ! "
		self.original.start(f"filesrc location={VIDEO} ! qtdemux ! h264parse ! avdec_h264 ! videoscale ! {SIZE} ! "
		                    f"{grey}videoconvert ! gtksink name=sink sync=true")
		self.status.set_text("Playing.")
		return False  # run once (GLib.idle_add)

	def stop_all(self, message=None):
		self.processed.stop()
		self.original.stop()
		if message:
			self.status.set_text(message)

	def tick(self):
		self.original.update_stats()
		self.processed.update_stats()
		return True

	def quit(self, *args):
		self.stop_all()
		Gtk.main_quit()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", default="172.20.10.10", help="iMX8's IP address")
	parser.add_argument("--port", type=int, default=5001, help="relay's listen port")
	parser.add_argument("--connect", action="store_true", help="connect straight away")
	args = parser.parse_args()
	Gst.init(None)
	window = DemoWindow(args.host, args.port)
	window.show_all()
	if args.connect:
		window.on_connect(None)
	Gtk.main()


if __name__ == "__main__":
	main()