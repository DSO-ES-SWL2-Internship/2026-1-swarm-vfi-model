
#!/usr/bin/env python3
"""Create a 24 FPS frame-held baseline using the relay's drop pattern (same probability, seed and max run)."""

import argparse
import subprocess
from pathlib import Path

from frame_dropper import RandomFrameDropper


WIDTH = 448
HEIGHT = 256
FPS = 24
FRAME_BYTES = WIDTH * HEIGHT


def read_exact(stream, size):
    """Read one complete raw frame, or return None at EOF."""
    chunks = []
    remaining = size

    while remaining:
        data = stream.read(remaining)
        if not data:
            if remaining == size:
                return None
            raise RuntimeError("Incomplete raw frame from FFmpeg")

        chunks.append(data)
        remaining -= len(data)

    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-probability", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-run", type=int, default=1,
                        help="longest burst of consecutive dropped frames (same as relay.py --max-drop-run)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")

    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")

    if not 0 <= args.drop_probability <= 1:
        parser.error("Drop probability must be between 0 and 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dropper = RandomFrameDropper(
        probability=args.drop_probability,
        seed=args.seed,
        max_run=args.max_run,
    )

    # Decode to model-sized grayscale frames.
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(args.input),
        "-map", "0:v:0", "-an",
        "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT},format=gray",
        "-pix_fmt", "gray",
        "-f", "rawvideo", "pipe:1",
    ]

    # Encode a constant-frame-rate 24 FPS baseline.
    encode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(FPS),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(args.output),
    ]

    decoder = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
    )

    encoder = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
    )

    previous_frame = None
    count = 0

    try:
        while True:
            frame = read_exact(decoder.stdout, FRAME_BYTES)
            if frame is None:
                break

            if dropper.should_drop():
                if previous_frame is None:
                    raise RuntimeError("First frame cannot be dropped")
                output_frame = previous_frame
            else:
                output_frame = frame
                previous_frame = frame

            encoder.stdin.write(output_frame)
            count += 1

        decoder.stdout.close()
        encoder.stdin.close()

        if decoder.wait() != 0:
            raise RuntimeError("FFmpeg decoder failed")

        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg encoder failed")

    except Exception:
        decoder.kill()
        encoder.kill()
        decoder.wait()
        encoder.wait()
        raise

    print(f"\nSaved: {args.output}")
    print(f"Output frames: {count}")
    dropper.report()


if __name__ == "__main__":
    main()
