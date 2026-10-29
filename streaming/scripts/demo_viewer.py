# Live demo window for the 3-device setup (runs on the VM, instead of receiver.py): can VFI recover the frames a
# stream has lost? Three panels, side by side (lossy next to recovered, for the comparison):
#   Lossy      the trailer with every other frame dropped (12 fps), as the Pi sends it
#   Recovered  the stream the iMX8 sends back after the two hops (Pi -> iMX8 -> VM): the lossy 12 fps with the
#              dropped frames interpolated by the relay's VFI, so 24 fps again
#   Original   the trailer as intended (24 fps)
# A "Show" checkbox per panel hides or shows it; all three are shown at the start.
#
# Lossy and Original are the VM's own copies of the videos, shown the way the model sees them -- scaled to
# 448x256 and in grey -- unless "Colour" is ticked. They start when the first processed frame arrives, so the
# panels run roughly together (the recovered side lags by the network and relay delay; it is not frame-exact).
# The recovered side shows frames as they arrive; the fps counters show 12 vs 24.
#
# The lossy video is made beforehand (not dropped live, which would add load and timing jitter):
# ./make_lossy_video.sh makes ../videos/sintel_trailer-480p-12fps.mp4; copy it to the Pi's videos folder too.
#
# Usage: start the sender on the Pi with the lossy video (./run_sender.sh --video sintel_trailer-480p-12fps.mp4)
# and the relay on the iMX8 as usual (run_relay.sh; it doubles the frame rate, 12 -> 24 fps), then
#   python3 demo_viewer.py [--host 172.20.10.10] [--port 5001] [--res 480p|256p]
# 256p: the Pi sends sintel_trailer-256p-12fps.mp4, already the model's 448x256, so the relay's scaler has nothing to do.
# and press Connect (or pass --connect). The relay exits when the video ends, so restart it before pressing Connect again.

import argparse
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gst, Gtk

VIDEOS = Path(__file__).resolve().parent.parent / "videos"
# Original and lossy (make_lossy_video.sh) per --res; 256p is already the model's 448x256, so the relay doesn't scale.
VIDEO_FILES = {res: (VIDEOS / f"sintel_trailer-{res}.mp4", VIDEOS / f"sintel_trailer-{res}-12fps.mp4")
               for res in ("480p", "256p")}
SIZE = "video/x-raw,width=448,height=256"


class Side:
	# One video panel: a GStreamer pipeline drawing into a gtksink widget, with a caption and an fps counter.
	def __init__(self, title, scale):
		self.title = title
		self.width, self.height = int(448 * scale), int(256 * scale)
		self.pipeline = None
		self.frames, self.window_start, self.window_frames = 0, time.time(), 0
		self.caption = Gtk.Label(label=title)
		self.stats = Gtk.Label(label="-")
		self.slot = Gtk.Box()
		self.slot.set_size_request(self.width, self.height)
		self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
		self.box.pack_start(self.caption, False, False, 0)
		self.box.pack_start(self.slot, True, True, 0)  # grows with the window
		self.box.pack_start(self.stats, False, False, 0)

	def start(self, description, on_first_frame=None):
		self.stop()
		self.pipeline = Gst.parse_launch(description)
		sink = self.pipeline.get_by_name("sink")
		self.slot.foreach(self.slot.remove)
		widget = sink.get_property("widget")
		widget.set_size_request(self.width, self.height)
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
	def __init__(self, host, port, scale, res):
		super().__init__(title="VFI demo: recovering lost frames, 12 -> 24 fps")
		self.connect("destroy", self.quit)
		self.lossy = Side("Lossy: every other frame dropped (12 fps)", scale)
		self.original = Side("Original (24 fps)", scale)
		self.processed = Side("Recovered by VFI: 12 -> 24 fps (Pi -> iMX8 -> VM)", scale)
		self.sides = [self.lossy, self.processed, self.original]
		self.original_video, self.lossy_video = VIDEO_FILES[res]

		# Add buttons to GUI
		self.host = Gtk.Entry(text=host, width_chars=15)
		self.port = Gtk.SpinButton.new_with_range(1, 65535, 1)
		self.port.set_value(port)
		self.colour = Gtk.CheckButton(label="Colour")
		connect = Gtk.Button(label="Connect")
		connect.connect("clicked", self.on_connect)
		stop = Gtk.Button(label="Stop")
		stop.connect("clicked", lambda button: self.stop_all("Stopped."))
		shows = []
		for side, name in zip(self.sides, ("Lossy", "Recovered", "Original")):
			show = Gtk.CheckButton(label=name, active=True)
			show.connect("toggled", self.on_show, side)
			shows.append(show)
		self.status = Gtk.Label(label="Start the sender and relay, then press Connect.", xalign=0)

		# Add video streams to GUI
		controls = Gtk.Box(spacing=8)
		for widget in (Gtk.Label(label="iMX8"), self.host, Gtk.Label(label="port"), self.port, self.colour, connect, stop,
		               Gtk.Label(label="   Show:"), *shows):
			controls.pack_start(widget, False, False, 0)
		videos = Gtk.Box(spacing=12, homogeneous=True)
		for side in self.sides:
			videos.pack_start(side.box, True, True, 0)
		layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
		layout.pack_start(controls, False, False, 0)
		layout.pack_start(videos, True, True, 0)
		layout.pack_start(self.status, False, False, 0)
		self.add(layout)
		GLib.timeout_add(500, self.tick)

	def on_show(self, button, side):
		# Hiding only hides the panel; its video keeps playing, so it is still in step when shown again.
		side.box.set_visible(button.get_active())
		self.resize(1, 1)  # shrink to fit the panels that are left

	def on_connect(self, button):
		self.stop_all()
		if not self.lossy_video.exists():
			self.status.set_text(f"{self.lossy_video.name} not found: run ./make_lossy_video.sh first.")
			return
		host, port = self.host.get_text().strip(), int(self.port.get_value())
		self.status.set_text(f"Connecting to {host}:{port}, waiting for the first frame...")
		# decodebin picks the decoder for whichever codec the relay sends (H.264 from the VPU, or MPEG-4).
		self.processed.start(f"tcpclientsrc host={host} port={port} ! tsdemux ! decodebin ! videoconvert ! "
		                     f"gtksink name=sink sync=false", on_first_frame=self.start_local)

	def start_local(self):
		# The VM's own copies of the lossy and original videos, started together with the first processed frame.
		grey = "" if self.colour.get_active() else "videoconvert ! video/x-raw,format=GRAY8 ! "
		for side, video in ((self.lossy, self.lossy_video), (self.original, self.original_video)):
			side.start(f"filesrc location={video} ! qtdemux ! h264parse ! avdec_h264 ! videoscale ! {SIZE} ! "
			           f"{grey}videoconvert ! gtksink name=sink sync=true")
		self.status.set_text("Playing.")
		return False  # run once (GLib.idle_add)

	def stop_all(self, message=None):
		for side in self.sides:
			side.stop()
		if message:
			self.status.set_text(message)

	def tick(self):
		for side in self.sides:
			side.update_stats()
		return True

	def quit(self, *args):
		self.stop_all()
		Gtk.main_quit()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", default="172.20.10.10", help="iMX8's IP address")
	parser.add_argument("--port", type=int, default=5001, help="relay's listen port")
	parser.add_argument("--scale", type=float, default=1.4, help="size of each panel, x 448x256 (window can be enlarged)")
	parser.add_argument("--res", choices=VIDEO_FILES, default="480p",
	                    help="the videos the Pi sends: 480p, or 256p (already 448x256, for the no-scaler relay)")
	parser.add_argument("--connect", action="store_true", help="connect straight away")
	args = parser.parse_args()
	Gst.init(None)
	window = DemoWindow(args.host, args.port, args.scale, args.res)
	window.show_all()
	if args.connect:
		window.on_connect(None)
	Gtk.main()


if __name__ == "__main__":
	main()
