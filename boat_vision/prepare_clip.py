from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a smaller MP4 sample from a large local video file.")
    parser.add_argument("--source", required=True, help="Large source video path.")
    parser.add_argument("--out", default="data/videos/sample.mp4", help="Output MP4 path.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--seconds", type=float, default=120.0, help="Output duration in seconds.")
    parser.add_argument("--max-width", type=int, default=1280, help="Downscale to this width if source is wider.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output FPS.")
    return parser.parse_args()


def scaled_size(width: int, height: int, max_width: int) -> Tuple[int, int]:
    if width <= max_width:
        return width, height
    scale = max_width / width
    return max_width, int(round(height * scale))


def read_source_fps(capture: cv2.VideoCapture) -> Optional[float]:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return float(fps)
    return None


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source video: {source_path}")

    source_fps = read_source_fps(capture) or args.fps
    start_frame = int(args.start * source_fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise SystemExit(f"Could not read from source video at {args.start} seconds")

    source_height, source_width = frame.shape[:2]
    out_width, out_height = scaled_size(source_width, source_height, args.max_width)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (out_width, out_height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not open output video: {output_path}")

    source_step = max(1, round(source_fps / args.fps))
    max_output_frames = int(args.seconds * args.fps)
    written = 0
    source_frame_index = start_frame

    try:
        while written < max_output_frames and ok:
            if (source_frame_index - start_frame) % source_step == 0:
                if (out_width, out_height) != (source_width, source_height):
                    frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1

            ok, frame = capture.read()
            source_frame_index += 1
    finally:
        writer.release()
        capture.release()

    print(
        f"Wrote {written} frames to {output_path} "
        f"({out_width}x{out_height}, {args.fps:.3f} FPS)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
