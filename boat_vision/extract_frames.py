from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from recorded camera video for annotation.")
    parser.add_argument("--source", required=True, help="Input video file.")
    parser.add_argument("--out", default="data/datasets/maritime/raw_frames", help="Output folder for images.")
    parser.add_argument("--every", type=int, default=30, help="Save one frame every N input frames.")
    parser.add_argument("--prefix", default="camera", help="Filename prefix, e.g. bow_camera.")
    parser.add_argument("--max-frames", type=int, help="Maximum saved frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    read_index = 0
    saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if read_index % args.every == 0:
            path = output_dir / f"{args.prefix}_{read_index:08d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            if args.max_frames is not None and saved >= args.max_frames:
                break
        read_index += 1

    capture.release()
    print(f"Saved {saved} frames to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
