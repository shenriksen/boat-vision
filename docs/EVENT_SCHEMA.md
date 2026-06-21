# Detection Event Schema

Events are newline-delimited JSON. One detection equals one JSON object.

Current schema version:

```text
boat_vision.detection.v1
```

Example:

```json
{"schema_version":"boat_vision.detection.v1","event_type":"object_detection","timestamp_utc":"2026-06-19T12:00:00+00:00","camera_id":"bow_camera","source":"data/videos/sample.mp4","frame_index":12,"video_time_sec":0.4,"model":"yolo26n.pt","class_id":8,"class_name":"boat","confidence":0.72,"tracking_id":null,"bbox_xyxy":{"x1":100.0,"y1":80.0,"x2":240.0,"y2":180.0},"bbox_xywh":{"x":100.0,"y":80.0,"width":140.0,"height":100.0},"image_size":{"width":1920,"height":1080},"vessel_pose":null}
```

Fields:

- `schema_version`: Stable schema identifier for downstream consumers.
- `event_type`: Currently `object_detection`.
- `timestamp_utc`: Wall-clock time when inference emitted the detection.
- `camera_id`: User-defined camera identifier.
- `source`: Input source with RTSP credentials masked.
- `frame_index`: Processed result index.
- `video_time_sec`: Approximate timestamp within a recorded video when FPS is known.
- `model`: Model name or checkpoint path.
- `class_id`: Numeric YOLO class ID.
- `class_name`: Human-readable class name.
- `confidence`: YOLO confidence score from 0.0 to 1.0.
- `tracking_id`: Reserved for future object tracking.
- `bbox_xyxy`: Bounding box corners in source image pixels.
- `bbox_xywh`: Bounding box origin/size in source image pixels.
- `image_size`: Source image dimensions in pixels.
- `vessel_pose`: Reserved for future GPS/heading/georeferencing metadata.
