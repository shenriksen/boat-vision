from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import cv2
from ultralytics import YOLO

from boat_vision.config import VisionConfig, apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local YOLO detections on a boat camera video source.")
    parser.add_argument("--config", default="configs/poc.yaml", help="Path to YAML configuration.")
    parser.add_argument("--source", help="Video file, webcam index, HTTP URL, or RTSP stream URL.")
    parser.add_argument("--camera-id", dest="camera_id", help="Camera identifier to include in JSON events.")
    parser.add_argument("--model", help="Ultralytics model path/name, e.g. yolo26n.pt or runs/.../best.pt.")
    parser.add_argument("--output-jsonl", dest="output_jsonl", help="Path for newline-delimited JSON detection events.")
    parser.add_argument("--conf", dest="confidence_threshold", type=float, help="Detection confidence threshold.")
    parser.add_argument("--iou", dest="iou_threshold", type=float, help="NMS IOU threshold.")
    parser.add_argument("--imgsz", dest="image_size", type=int, help="Inference image size.")
    parser.add_argument("--vid-stride", dest="frame_stride", type=int, help="Process every Nth video frame.")
    parser.add_argument("--max-frames", dest="max_frames", type=int, help="Stop after this many result frames.")
    parser.add_argument("--device", help="Device such as cpu, mps, or 0.")
    parser.add_argument("--show", dest="display", action="store_true", help="Show annotated frames in a window.")
    parser.add_argument("--no-annotated-video", dest="save_annotated_video", action="store_false")
    parser.set_defaults(save_annotated_video=None)
    return parser.parse_args()


def masked_source(source: str) -> str:
    return re.sub(r"(rtsp://)([^:/@\s]+):([^@\s]+)@", r"\1***:***@", source)


def normalize_source(source: str) -> Union[str, int]:
    if source.isdigit():
        return int(source)
    return source


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def detect_source_fps(source: Union[str, int]) -> Optional[float]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps and fps > 0:
        return float(fps)
    return None


def frame_time_sec(frame_index: int, config: VisionConfig, fps: Optional[float]) -> Optional[float]:
    if not fps:
        return None
    return round((frame_index * config.frame_stride) / fps, 3)


def class_name_for(result: Any, class_id: int) -> str:
    names = result.names or {}
    return str(names.get(class_id, class_id))


def iter_detection_events(
    result: Any,
    config: VisionConfig,
    frame_index: int,
    fps: Optional[float],
) -> Iterable[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return

    height, width = result.orig_shape
    allowed = set(config.allowed_classes or [])
    xyxy_values = boxes.xyxy.cpu().tolist()
    conf_values = boxes.conf.cpu().tolist()
    cls_values = boxes.cls.cpu().tolist()

    for xyxy, confidence, class_value in zip(xyxy_values, conf_values, cls_values):
        class_id = int(class_value)
        class_name = class_name_for(result, class_id)
        if allowed and class_name not in allowed:
            continue

        x1, y1, x2, y2 = [float(value) for value in xyxy]
        yield {
            "schema_version": "boat_vision.detection.v1",
            "event_type": "object_detection",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "camera_id": config.camera_id,
            "source": masked_source(config.source),
            "frame_index": frame_index,
            "video_time_sec": frame_time_sec(frame_index, config, fps),
            "model": config.model,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(confidence),
            "tracking_id": None,
            "bbox_xyxy": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "bbox_xywh": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            },
            "image_size": {
                "width": int(width),
                "height": int(height),
            },
            "vessel_pose": None,
        }


def run_detection(config: VisionConfig) -> int:
    ensure_parent(config.output_jsonl)
    if config.save_annotated_video:
        ensure_parent(config.annotated_video_path)

    model = YOLO(config.model)
    source = normalize_source(config.source)
    fps = config.source_fps or detect_source_fps(source)
    predict_kwargs: Dict[str, Any] = {
        "source": source,
        "stream": True,
        "conf": config.confidence_threshold,
        "iou": config.iou_threshold,
        "imgsz": config.image_size,
        "vid_stride": config.frame_stride,
        "stream_buffer": config.stream_buffer,
        "verbose": False,
    }
    if config.device not in (None, ""):
        predict_kwargs["device"] = config.device

    writer: Optional[cv2.VideoWriter] = None
    event_count = 0
    frame_count = 0

    event_mode = "a" if config.append_events else "w"
    with Path(config.output_jsonl).open(event_mode, encoding="utf-8") as event_file:
        for frame_index, result in enumerate(model.predict(**predict_kwargs)):
            frame_count += 1

            for event in iter_detection_events(result, config, frame_index, fps):
                event_file.write(json.dumps(event, separators=(",", ":")) + "\n")
                event_count += 1

            if config.save_annotated_video or config.display:
                annotated = result.plot()
                if config.save_annotated_video:
                    if writer is None:
                        height, width = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            config.annotated_video_path,
                            fourcc,
                            float(config.annotated_video_fps),
                            (width, height),
                        )
                    writer.write(annotated)
                if config.display:
                    cv2.imshow(f"{config.camera_id} detections", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if config.max_frames is not None and frame_count >= config.max_frames:
                break

    if writer is not None:
        writer.release()
    if config.display:
        cv2.destroyAllWindows()

    print(
        f"Processed {frame_count} frames from {masked_source(config.source)}; "
        f"wrote {event_count} detection events to {config.output_jsonl}."
    )
    if fps:
        print(f"Source FPS used for video_time_sec: {fps:.3f}")
    if config.save_annotated_video:
        print(f"Annotated video: {config.annotated_video_path}")
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(
        config,
        {
            "source": args.source,
            "camera_id": args.camera_id,
            "model": args.model,
            "output_jsonl": args.output_jsonl,
            "confidence_threshold": args.confidence_threshold,
            "iou_threshold": args.iou_threshold,
            "image_size": args.image_size,
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "device": args.device,
            "display": args.display if args.display else None,
            "save_annotated_video": args.save_annotated_video,
        },
    )
    return run_detection(config)


if __name__ == "__main__":
    raise SystemExit(main())
