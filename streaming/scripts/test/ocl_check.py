# Can the GPU (imxvideoconvert_ocl, OpenCL) scale the VPU's frames for the relay, so that no grey conversion is left
# for the CPU? The G2D outputs RGB only, so on the G2D chain the CPU converts RGB -> grey and then looks up the video
# range. imxvideoconvert_ocl takes the VPU's NV12 and outputs NV12, whose Y plane is already the video-range grey the
# model takes.
#
# Run on the iMX8: python3 streaming/scripts/test/ocl_check.py [--repeats 2] [--frame 700] [--frames 1251]
#
# 1. Speed: the 480p trailer flat out (fakesink sync=false), after vpudec: decode alone, each scaler (OCL, G2D, CPU
#    videoscale) alone, and each followed by videoconvert to the GRAY8 the relay's appsink takes. Wall time and CPU time
#    (whole board, /proc/stat) per frame, fastest of --repeats runs, start-up included.
# 2. Correctness: one frame through each scaler, compared with a reference that never touches VPU memory (avdec_h264 ->
#    videoscale). Checks the size and stride (a stride ignored would shear the picture), the Y range (video range is
#    16-235), and the difference from the reference, also with the frame shifted by up to 2 pixels (a shift would show
#    as a much smaller difference at some offset). The G2D's grey is full range, so it is mapped to video range first.

import argparse

import gi
import numpy as np
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

from pipeline_bench import VIDEOS, run

SIZE = "width=448,height=256"
DECODE = "filesrc location={video} ! qtdemux ! h264parse ! vpudec frame-drop=false"
SCALERS = {
	"ocl": f"imxvideoconvert_ocl ! video/x-raw,format=NV12,{SIZE}",
	"g2d": f"imxvideoconvert_g2d ! video/x-raw,{SIZE}",
	"cpu": f"videoscale ! video/x-raw,{SIZE}",
}
GREY = "videoconvert ! video/x-raw,format=GRAY8"
# Frame check: the reference decodes and scales on the CPU only; the G2D's grey uses the relay's colorimetry.
FRAME_PIPELINES = {
	"ref": f"avdec_h264 ! videoscale ! video/x-raw,format=I420,{SIZE}",
	"ocl": f"vpudec frame-drop=false ! {SCALERS['ocl']}",
	"g2d": f"vpudec frame-drop=false ! {SCALERS['g2d']} ! {GREY},colorimetry=2:4:0:0",
	"cpu": f"vpudec frame-drop=false ! {SCALERS['cpu']}",
}


def speed(video, repeats, frames):
	print(f"1. Speed: {video.name}, {frames} frames, flat out, fastest of {repeats}; ms per frame (start-up included)")
	print(f"{'after vpudec':40} {'wall':>6} {'CPU':>6}")
	rows = [("decode alone", "")]
	for name, scaler in SCALERS.items():
		rows += [(f"{name} scale", f" ! {scaler}"), (f"{name} scale + grey", f" ! {scaler} ! {GREY}")]
	rows.append(("no scale + grey (full 480p)", f" ! {GREY}"))
	for name, tail in rows:
		wall, cpu = min(run(DECODE.format(video=video) + tail + " ! fakesink sync=false") for _ in range(repeats))
		print(f"{name:40} {wall / frames * 1000:6.2f} {cpu / frames * 1000:6.2f}", flush=True)


def grab(video, description, n):
	# The n-th frame's Y plane (448x256), read with the buffer's own stride and offset.
	pipe = Gst.parse_launch(f"filesrc location={video} ! qtdemux ! h264parse ! {description} ! "
	                        f"appsink name=sink sync=false max-buffers=1")
	sink = pipe.get_by_name("sink")
	pipe.set_state(Gst.State.PLAYING)
	for _ in range(n):
		sample = sink.emit("pull-sample")
	buf = sample.get_buffer()
	meta = GstVideo.buffer_get_video_meta(buf)
	if meta:
		offset, stride = meta.offset[0], meta.stride[0]
	else:
		offset, stride = 0, GstVideo.VideoInfo.new_from_caps(sample.get_caps()).stride[0]
	ok, info = buf.map(Gst.MapFlags.READ)
	y = np.frombuffer(info.data, np.uint8)[offset:offset + stride * 256].reshape(256, stride)[:, :448].astype(int)
	buf.unmap(info)
	pipe.set_state(Gst.State.NULL)
	return y, stride, buf.get_size(), meta is not None


def correctness(video, n):
	print(f"\n2. Correctness: frame {n} of {video.name}, Y plane 448x256")
	frames = {}
	for name, description in FRAME_PIPELINES.items():
		y, stride, size, has_meta = grab(video, description, n)
		p1, p99 = np.percentile(y, [1, 99]).astype(int)
		print(f"{name}: stride {stride} (video meta {'yes' if has_meta else 'no'}), {size} bytes; "
		      f"Y min {y.min()} p1 {p1} p99 {p99} max {y.max()} mean {y.mean():.1f}")
		frames[name] = np.round(16 + y * 219 / 255) if name == "g2d" else y
	ref = frames.pop("ref")
	inner = (slice(4, -4), slice(4, -4))
	for name, y in frames.items():
		diff = np.abs(y - ref)
		shifts = min((np.abs(np.roll(y, (dy, dx), (0, 1))[inner] - ref[inner]).mean(), dy, dx)
		             for dy in range(-2, 3) for dx in range(-2, 3))
		print(f"{name} vs ref: mean |diff| {diff.mean():.2f} (top 20 rows {diff[:20].mean():.2f}, bottom 20 "
		      f"{diff[-20:].mean():.2f}), mean diff {(y - ref).mean():+.2f}; best shift dy {shifts[1]} dx {shifts[2]} "
		      f"-> {shifts[0]:.2f}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--repeats", type=int, default=2)
	parser.add_argument("--frame", type=int, default=700, help="frame to compare (a bright one)")
	parser.add_argument("--frames", type=int, default=1251, help="frames in the video (as in pipeline_bench.py)")
	args = parser.parse_args()
	Gst.init(None)
	video = VIDEOS / "sintel_trailer-480p.mp4"
	speed(video, args.repeats, args.frames)
	correctness(video, args.frame)


if __name__ == "__main__":
	main()
