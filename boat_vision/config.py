from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class VisionConfig:
    model: str = "yolo26n.pt"
    source: str = "data/videos/sample.mp4"
    source_fps: Optional[float] = None
    camera_id: str = "bow_camera"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    frame_stride: int = 1
    max_frames: Optional[int] = None
    allowed_classes: Optional[list[str]] = None
    output_jsonl: str = "outputs/events/poc_detections.jsonl"
    append_events: bool = False
    save_annotated_video: bool = True
    annotated_video_path: str = "outputs/annotated/poc_annotated.mp4"
    annotated_video_fps: float = 15.0
    display: bool = False
    stream_buffer: bool = False
    device: Optional[Union[str, int]] = None


def load_config(path: str | Path) -> VisionConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    valid_keys = {field.name for field in fields(VisionConfig)}
    unknown = sorted(set(data) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown)}")
    return VisionConfig(**data)


def apply_overrides(config: VisionConfig, overrides: dict[str, Any]) -> VisionConfig:
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config
