#!/usr/bin/env python3
"""Three-device VFI demo viewer (VM).

The recovered panel is LIVE MPEG-TS/TCP from the i.MX8. Baseline and original
panels are local MP4 files; all panels are displayed in grayscale by default.
This viewer does not configure the Pi or relay and is approximately, not
frame-exactly, synchronized with the live network stream.
"""
import argparse
import time
from pathlib import Path

import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, Gst, Gtk

VIDEOS = Path(__file__).resolve().parent.parent / 'videos'
SIZE = 'video/x-raw,width=448,height=256'
MODES = {
    'original': ('Original 12 to 24 FPS', 'sintel_trailer-{res}-12fps.mp4',
                 'sintel_trailer-{res}.mp4', '12 FPS input'),
    'random': ('Random isolated frame-loss recovery', 'stuttering_p50.mp4',
               'sintel_trailer-480p.mp4', 'Frame-held 24 FPS baseline'),
    'burst': ('Burst frame-loss recovery', 'stuttering_burst.mp4',
              'sintel_trailer-480p.mp4', 'Frame-held 24 FPS baseline'),
    'recursive': ('Recursive 6 to 24 FPS', 'framehold_6to24.mp4',
                  'sintel_trailer-480p.mp4', '6 FPS frame-held at 24 FPS'),
}


def quote_gst(value):
    # Escape string property values in Gst.parse_launch descriptions.
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


class Side:
    def __init__(self, title, scale):
        self.width, self.height = int(448 * scale), int(256 * scale)
        self.pipeline = None
        self.frames = self.window_frames = 0
        self.window_start = time.monotonic()
        self.first_frame_callback = None
        self.caption = Gtk.Label(label=title)
        self.stats = Gtk.Label(label='-')
        self.slot = Gtk.Box()
        self.slot.set_size_request(self.width, self.height)
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.box.pack_start(self.caption, False, False, 0)
        self.box.pack_start(self.slot, True, True, 0)
        self.box.pack_start(self.stats, False, False, 0)

    def start(self, description, on_first_frame=None):
        self.stop()
        self.pipeline = Gst.parse_launch(description)
        sink = self.pipeline.get_by_name('sink')
        if sink is None:
            raise RuntimeError('GStreamer pipeline has no sink named sink')
        for child in self.slot.get_children():
            self.slot.remove(child)
        widget = sink.get_property('widget')
        widget.set_size_request(self.width, self.height)
        self.slot.pack_start(widget, True, True, 0)
        widget.show()
        self.frames = self.window_frames = 0
        self.window_start = time.monotonic()
        self.first_frame_callback = on_first_frame
        sink.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, self.on_frame)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::error', self.on_error)
        bus.connect('message::eos', self.on_eos)
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Could not start GStreamer pipeline')

    def on_frame(self, pad, info):
        self.frames += 1
        self.window_frames += 1
        if self.frames == 1 and self.first_frame_callback:
            GLib.idle_add(self.first_frame_callback)
        return Gst.PadProbeReturn.OK

    def on_eos(self, bus, msg):
        self.stats.set_text(f'{self.frames} frames - finished')

    def on_error(self, bus, msg):
        err, debug = msg.parse_error()
        self.stats.set_text(f'Error: {err.message}')
        print(f'{self.caption.get_text()}: {err.message}\n{debug or ""}')
        self.stop()

    def update_stats(self):
        now = time.monotonic()
        elapsed = now - self.window_start
        if self.pipeline and elapsed >= 1:
            self.stats.set_text(f'{self.frames} frames, {self.window_frames / elapsed:.1f} fps')
            self.window_frames = 0
            self.window_start = now

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.get_bus().remove_signal_watch()
            self.pipeline = None


class DemoWindow(Gtk.Window):
    def __init__(self, args):
        super().__init__(title='VFI demonstration - Pi to i.MX8 to VM')
        self.connect('destroy', self.quit)
        self.args = args
        self.baseline = Side('Without VFI', args.scale)
        self.recovered = Side('With VFI', args.scale)
        self.original = Side('Original 24 FPS reference', args.scale)
        self.sides = [self.baseline, self.recovered, self.original]
        self.panel_spacing = 12
        self.window_margin = 12
        self.host = Gtk.Entry(text=args.host, width_chars=15)
        self.port = Gtk.SpinButton.new_with_range(1, 65535, 1)
        self.port.set_value(args.port)
        self.mode = Gtk.ComboBoxText()
        for key, (title, *_rest) in MODES.items():
            self.mode.append(key, title)
        self.mode.set_active_id(args.mode)
        self.mode.connect('changed', self.on_mode_changed)
        self.colour = Gtk.CheckButton(label='Colour (disable grayscale)')
        self.colour.set_active(args.colour)
        self.colour.connect('toggled', self.on_colour_changed)
        connect = Gtk.Button(label='Connect')
        connect.connect('clicked', self.on_connect)
        stop = Gtk.Button(label='Stop')
        stop.connect('clicked', lambda _b: self.stop_all('Stopped.'))
        self.status = Gtk.Label(label='Start Pi sender and i.MX8 relay, then Connect.', xalign=0)
        self.status.set_line_wrap(True)
        controls = Gtk.Box(spacing=8)
        for item in (Gtk.Label(label='Mode'), self.mode, Gtk.Label(label='i.MX8'),
                     self.host, Gtk.Label(label='Port'), self.port, self.colour, connect, stop):
            controls.pack_start(item, False, False, 0)
        shows = Gtk.Box(spacing=8)
        shows.pack_start(Gtk.Label(label='Show:'), False, False, 0)
        for side, label in zip(self.sides, ('Baseline', 'Recovered', 'Original')):
            check = Gtk.CheckButton(label=label)
            check.set_active(label != 'Original')
            check.connect('toggled', self.on_show, side)
            shows.pack_start(check, False, False, 0)
            side.box.set_visible(label != 'Original')
        # Fixed-width panels: hiding one must not stretch the others.
        panels = Gtk.Box(spacing=self.panel_spacing, homogeneous=False)
        panels.set_halign(Gtk.Align.CENTER)
        for side in self.sides:
            side.box.set_size_request(side.width, -1)
            panels.pack_start(side.box, False, False, 0)
        layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
        for item, expand in ((controls, False), (shows, False), (panels, True), (self.status, False)):
            layout.pack_start(item, expand, expand, 0)
        self.add(layout)
        self.update_titles()
        GLib.timeout_add(500, self.tick)
        GLib.idle_add(self.fit_window_to_panels)

    def selected_files(self):
        mode = self.mode.get_active_id()
        _title, baseline, original, _label = MODES[mode]
        return (VIDEOS / baseline.format(res=self.args.res),
                VIDEOS / original.format(res=self.args.res))

    def update_titles(self):
        mode = self.mode.get_active_id()
        title, _baseline, _original, label = MODES[mode]
        self.set_title(f'VFI demo: {title}')
        self.baseline.caption.set_text('Without VFI')
        self.recovered.caption.set_text('With VFI')
        self.original.caption.set_text('Original 24 FPS reference')

    def on_mode_changed(self, _widget):
        self.stop_all('Mode changed. Configure sender/relay for this mode, then Connect.')
        self.update_titles()

    def on_colour_changed(self, _widget):
        self.stop_all('Display colour changed. Press Connect to restart.')

    def fit_window_to_panels(self):
        """Fit window to visible fixed-aspect panels, subject to control-bar width."""
        visible = sum(side.box.get_visible() for side in self.sides)
        if not visible:
            visible = 1
        width = (visible * self.baseline.width +
                 (visible - 1) * self.panel_spacing +
                 2 * self.window_margin)
        # GTK may impose a wider minimum due to the controls; panels stay fixed.
        self.resize(width, self.baseline.height + 135)
        return False

    def on_show(self, button, side):
        side.box.set_visible(button.get_active())
        GLib.idle_add(self.fit_window_to_panels)

    def display_chain(self):
        # Keep grayscale conversion after decoding, on ALL panels.
        if self.colour.get_active():
            return f'videoconvert ! videoscale ! {SIZE} ! videoconvert ! '
        return (f'videoconvert ! videoscale ! {SIZE} ! '
                'videoconvert ! video/x-raw,format=GRAY8 ! videoconvert ! ')

    def on_connect(self, _button):
        self.stop_all()
        baseline, original = self.selected_files()
        missing = [str(p) for p in (baseline, original) if not p.is_file()]
        if missing:
            self.status.set_text('Missing local video(s): ' + ', '.join(missing))
            return
        host = self.host.get_text().strip()
        port = int(self.port.get_value())
        if not host:
            self.status.set_text('Enter the i.MX8 host address.')
            return
        self.status.set_text(f'Connecting to {host}:{port}; waiting for recovered frames...')
        # The VM connects OUTBOUND to the relay's TCP server; no UDP needed.
        pipeline = (f'tcpclientsrc host={quote_gst(host)} port={port} ! '
                    f'tsdemux ! decodebin ! {self.display_chain()}'
                    'gtksink name=sink sync=false')
        try:
            self.recovered.start(pipeline, on_first_frame=self.start_local)
        except (RuntimeError, GLib.Error) as exc:
            self.status.set_text(f'Unable to start live stream: {exc}')

    def start_local(self):
        # Triggered on first recovered frame. This is approximate synchronization,
        # not frame-accurate alignment to the network stream's original PTS.
        baseline, original = self.selected_files()
        try:
            for side, path in ((self.baseline, baseline), (self.original, original)):
                pipeline = (f'filesrc location={quote_gst(path)} ! '
                            f'decodebin ! {self.display_chain()}'
                            'gtksink name=sink sync=true')
                side.start(pipeline)
            self.status.set_text('Playing.')
        except (RuntimeError, GLib.Error) as exc:
            self.stop_all(f'Unable to start local video: {exc}')
        return False

    def stop_all(self, message=None):
        for side in self.sides:
            side.stop()
        if message:
            self.status.set_text(message)

    def tick(self):
        for side in self.sides:
            side.update_stats()
        return True

    def quit(self, *_args):
        self.stop_all()
        Gtk.main_quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='172.20.10.10')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--scale', type=float, default=1.2)
    parser.add_argument('--res', choices=['480p', '256p'], default='480p',
                        help='Source resolution for the original 12-to-24 mode')
    parser.add_argument('--mode', choices=list(MODES), default='random')
    parser.add_argument('--colour', action='store_true', help='Show colour instead of grayscale')
    parser.add_argument('--connect', action='store_true', help='Connect at startup')
    args = parser.parse_args()
    Gst.init(None)
    window = DemoWindow(args)
    window.show_all()
    window.original.box.hide()  # third panel is optional by default
    GLib.idle_add(window.fit_window_to_panels)
    if args.connect:
        GLib.idle_add(window.on_connect, None)
    Gtk.main()

if __name__ == '__main__':
    main()
