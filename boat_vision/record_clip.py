from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Union

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a short MP4 clip from an RTSP/HTTP camera stream.")
    parser.add_argument("--source", required=True, help="RTSP/HTTP stream URL, video file, or webcam index.")
    parser.add_argument("--out", default="data/videos/camera_clip.mp4", help="Output MP4 path.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Recording duration.")
    parser.add_argument("--fps", type=float, help="Output FPS override if the stream does not report FPS.")
    parser.add_argument("--camera-id", default="camera", help="Used only in progress output.")
    return parser.parse_args()


def normalize_source(source: str) -> Union[str, int]:
    if source.isdigit():
        return int(source)
    return source


def masked_source(source: str) -> str:
    return re.sub(r"(rtsp://)([^:/@\s]+):([^@\s]+)@", r"\1***:***@", source)


def main() -> int:
    args = parse_args()
    source = normalize_source(args.source)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {masked_source(args.source)}")

    fps = args.fps or capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise SystemExit(f"Could not read first frame from: {masked_source(args.source)}")
        height, width = frame.shape[:2]
        first_frame = frame
    else:
        first_frame = None

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video writer: {output_path}")

    print(f"Recording {args.camera_id} from {masked_source(args.source)} to {output_path}")
    start = time.monotonic()
    frames = 0
    try:
        if first_frame is not None:
            writer.write(first_frame)
            frames += 1
        while time.monotonic() - start < args.seconds:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        writer.release()
        capture.release()

    print(f"Saved {frames} frames at {fps:.3f} FPS to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
