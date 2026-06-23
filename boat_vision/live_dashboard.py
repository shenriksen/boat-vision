from __future__ import annotations

import argparse
import html
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Use TCP for RTSP (more reliable than the default UDP across networks). We avoid
# aggressive low-latency flags here because they can stop some cameras from
# opening at all (VLC works because it uses sane defaults).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

MARITIME_CLASSES = [
    "boat",
    "ship",
    "sailboat",
    "small_vessel",
    "kayak",
    "navigation_buoy",
    "red_lateral_mark",
    "green_lateral_mark",
    "cardinal_mark",
    "special_mark",
    "floating_object",
    "person",
]


# Maps any model's class label (generic COCO, the Singapore maritime model, or a
# custom model) to Boat Vision's internal class names, so display/markers/filters
# work regardless of which model is loaded.
CLASS_ALIASES = {
    "boat": "boat", "ship": "ship", "sailboat": "sailboat", "sail boat": "sailboat",
    "small_vessel": "small_vessel", "kayak": "kayak",
    "speed boat": "small_vessel", "speedboat": "small_vessel",
    "buoy": "navigation_buoy", "navigation buoy": "navigation_buoy",
    "navigation_buoy": "navigation_buoy", "navigation mark": "navigation_buoy",
    "sea mark": "navigation_buoy", "seamark": "navigation_buoy", "lighthouse": "navigation_buoy",
    "ferry": "ship", "vessel-ship": "ship", "vessel": "ship",
    "person": "person", "swimmer": "person",
    "floating_object": "floating_object", "floating object": "floating_object",
    "other": "floating_object", "flying bird-plane": "floating_object",
    "red_lateral_mark": "red_lateral_mark", "green_lateral_mark": "green_lateral_mark",
    "cardinal_mark": "cardinal_mark", "special_mark": "special_mark",
}


def normalize_class(name: str) -> str:
    return CLASS_ALIASES.get(str(name).strip().lower(), str(name))


def masked_source(source: str) -> str:
    return re.sub(r"(rtsp://)([^:/@\s]+):([^@\s]+)@", r"\1***:***@", source)


def class_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
        return [item for item in values if item]
    return [str(item) for item in value if str(item).strip()]


def is_live_source(source: str) -> bool:
    lowered = source.lower()
    return source.isdigit() or lowered.startswith(("rtsp://", "http://", "https://"))


# --- AI-only image enhancement -------------------------------------------------
# A single fixed setting can't cope with sun, dusk, rain and haze, so the AI
# auto-levels exposure and rotates through several condition-tuned profiles
# (one per frame). An object invisible under one profile pops under another, and
# an area is only "clear" once it's been checked under all of them. None of this
# is ever shown to the operator — it feeds inference only.
ENHANCE_PROFILES = ("neutral", "low_light", "anti_glare", "haze_rain")


def _apply_gamma(img: Any, g: float) -> Any:
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _auto_level(img: Any) -> Any:
    """Normalise exposure via a percentile stretch so dark, bright and hazy frames
    all land in a consistent range before the per-condition profile is applied."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    if hi - lo < 10:
        return img
    return np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _boost_saturation(img: Any, factor: float) -> Any:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _clahe(img: Any, clip: float) -> Any:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _dehaze(img: Any) -> Any:
    """Subtract the atmospheric 'veil' (smoothed dark channel) to cut rain/fog/glare haze."""
    veil = cv2.min(cv2.min(img[..., 0], img[..., 1]), img[..., 2])
    veil = cv2.GaussianBlur(veil, (0, 0), 20)
    return np.clip(img.astype(np.int16) - 0.6 * veil[..., None], 0, 255).astype(np.uint8)


def enhance_for_profile(frame: Any, profile: str) -> Any:
    """Auto-level then apply one condition-tuned enhancement profile. AI-input only."""
    img = _auto_level(frame)
    if profile == "low_light":      # dusk / night: brighten + strong local contrast
        return _clahe(_boost_saturation(_apply_gamma(img, 0.55), 1.5), 3.0)
    if profile == "anti_glare":     # bright sun: pull down highlights, lift colour
        return _clahe(_boost_saturation(_apply_gamma(img, 1.4), 1.3), 2.0)
    if profile == "haze_rain":      # fog / rain / spray: dehaze + heavy contrast
        return _clahe(_boost_saturation(_dehaze(img), 1.6), 4.0)
    return _clahe(_boost_saturation(img, 1.4), 2.0)  # neutral daylight


class ObjectTracker:
    """Lightweight per-camera multi-object tracker. Gives the live-feed benefits of
    ByteTrack/BoT-SORT — stable IDs, no flickering boxes, tolerance of short occlusions
    — but fits our shared-model pipeline (Ultralytics' built-in tracker keeps state
    inside the model, which would collide across cameras). It works on detection
    *event dicts* from any source (YOLO or the colour buoy radar), with:
      - confidence hysteresis: a HIGH bar to START a track, a LOW bar to KEEP it;
      - N-of-M confirmation before a track is reported (kills one-frame false hits);
      - a coast buffer so a briefly-lost object isn't dropped mid-occlusion.
    """

    def __init__(self, start_conf: float, keep_conf: float, confirm_hits: int,
                 confirm_window: int, buffer_frames: int, match_dist: float) -> None:
        self.start_conf = start_conf
        self.keep_conf = keep_conf
        self.confirm_hits = max(1, confirm_hits)
        self.confirm_window = max(self.confirm_hits, confirm_window)
        self.buffer = max(0, buffer_frames)
        self.match_dist = match_dist
        self.tracks: List[Dict[str, Any]] = []
        self._next_id = 1

    @staticmethod
    def _center(ev: Dict[str, Any]) -> tuple[float, float]:
        b = ev["bbox_xyxy"]
        return (b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0

    def update(self, dets: List[Dict[str, Any]], step: int) -> List[Dict[str, Any]]:
        width = dets[0]["image_size"]["width"] if dets else 1
        tol = self.match_dist * max(1, width) if self.match_dist <= 1 else self.match_dist
        used: set[int] = set()

        # 1) Sustain existing tracks: match to the nearest same-class detection that
        #    clears the LOW (keep) threshold — easy to hold, hard to start.
        for t in self.tracks:
            tcx, tcy = self._center(t["event"])
            best, best_d = None, 1e18
            for i, ev in enumerate(dets):
                if i in used or ev["class_name"] != t["class_name"] or ev["confidence"] < self.keep_conf:
                    continue
                cx, cy = self._center(ev)
                if abs(cx - tcx) <= tol and abs(cy - tcy) <= tol:
                    d = abs(cx - tcx) + abs(cy - tcy)
                    if d < best_d:
                        best, best_d = i, d
            if best is not None:
                used.add(best)
                t["event"] = dets[best]
                t["misses"] = 0
                t["hits"].append(step)
                t["last"] = step
            else:
                t["misses"] += 1

        # 2) Start new tracks only from strong, unmatched detections (HIGH threshold).
        for i, ev in enumerate(dets):
            if i in used or ev["confidence"] < self.start_conf:
                continue
            self.tracks.append({
                "id": self._next_id, "class_name": ev["class_name"], "event": ev,
                "hits": [step], "misses": 0, "last": step, "confirmed": False,
            })
            self._next_id += 1

        # 3) Retire dead tracks; trim hit history to the confirmation window.
        alive = []
        for t in self.tracks:
            if t["misses"] > self.buffer:
                continue
            t["hits"] = [h for h in t["hits"] if step - h < self.confirm_window]
            if len(t["hits"]) >= self.confirm_hits:
                t["confirmed"] = True  # sticky: stays reported while it coasts
            alive.append(t)
        self.tracks = alive

        # 4) Report confirmed tracks (whether currently seen or coasting through occlusion).
        out = []
        for t in self.tracks:
            if not t["confirmed"]:
                continue
            ev = dict(t["event"])
            ev["tracking_id"] = t["id"]
            ev["coasting"] = t["misses"] > 0
            out.append(ev)
        return out


class PyAvCapture:
    """A cv2.VideoCapture-compatible reader backed by PyAV.

    PyAV's FFmpeg handles RTSP authentication like VLC does, so it connects to
    cameras that OpenCV's RTSP auth rejects with 401. Exposes the small subset
    of the VideoCapture API the workers use (isOpened/read/set/release).
    """

    def __init__(self, url: str) -> None:
        self._container = None
        self._frames = None
        try:
            import av  # noqa: PLC0415

            self._container = av.open(
                url,
                options={
                    "rtsp_transport": "tcp",
                    "stimeout": "10000000",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                    "max_delay": "200000",
                },
                timeout=12,
            )
            stream = self._container.streams.video[0]
            stream.thread_type = "AUTO"
            self._frames = self._container.decode(stream)
        except Exception as exc:  # noqa: BLE001
            print(f"PyAV could not open {masked_source(url)}: {exc}")
            self.release()

    def isOpened(self) -> bool:
        return self._frames is not None

    def read(self):
        if self._frames is None:
            return False, None
        try:
            frame = next(self._frames)
            return True, frame.to_ndarray(format="bgr24")
        except Exception:
            return False, None

    def set(self, *args, **kwargs) -> bool:
        return True

    def release(self) -> None:
        try:
            if self._container is not None:
                self._container.close()
        except Exception:
            pass
        self._container = None
        self._frames = None


def open_capture(source: str) -> Any:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    # RTSP: prefer PyAV (VLC-like auth). Fall back to OpenCV/FFmpeg if it fails.
    if source.lower().startswith("rtsp://"):
        av_cap = PyAvCapture(source)
        if av_cap.isOpened():
            return av_cap
    # HTTP/file (or RTSP fallback): OpenCV FFmpeg with open/read timeouts.
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    for prop, value in (
        (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None), 10000),
        (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None), 10000),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
    ):
        if prop is not None:
            try:
                capture.set(prop, value)
            except Exception:
                pass
    return capture


def load_yolo_model(model_path: str) -> tuple[YOLO, str, bool]:
    if model_path.endswith(".pt") and not Path(model_path).exists():
        fallback = "yolo26s.pt" if Path("yolo26s.pt").exists() else "yolo26n.pt"
        print(f"Configured model not found: {model_path}. Falling back to {fallback}.")
        return YOLO(fallback), fallback, True
    return YOLO(model_path), model_path, False


def resolve_device(requested: Optional[str]) -> str:
    """Return a device that actually works.

    torch.cuda.is_available() can return True on a machine whose GPU build still
    can't launch kernels (e.g. 'no kernel image is available for execution on the
    device' / driver mismatch). We verify a real kernel launch and fall back to
    CPU so the app always runs.
    """
    req = "" if requested is None else str(requested).strip()
    if req == "cpu":
        return "cpu"
    try:
        import torch  # noqa: PLC0415

        if req in ("", "auto"):
            req = "0" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            return "mps" if ok else "cpu"
        if req.isdigit():
            if not torch.cuda.is_available():
                print("CUDA not available; using CPU.")
                return "cpu"
            probe = torch.zeros(16, device=f"cuda:{req}")
            float((probe + 1).sum().item())  # forces a kernel launch
            return req
    except Exception as exc:  # noqa: BLE001
        print(f"GPU not usable ({exc}); falling back to CPU.")
        return "cpu"
    return req or "cpu"


@dataclass
class CameraConfig:
    camera_id: str
    name: str
    source: str
    enabled: bool
    allowed_classes: Optional[List[str]]
    ignore_zones: List[Dict[str, float]]
    ignore_polygons: List[List[Dict[str, float]]] = None
    detect: bool = True  # per-camera AI on/off (independent of the global toggle)


@dataclass
class DashboardConfig:
    columns: str
    show_events: bool
    card_min_width: int
    view_mode: str
    show_labels: bool = True
    show_status: bool = True
    show_vessel_area: bool = True


@dataclass
class AppConfig:
    host: str
    port: int
    model: str
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    device: Optional[str]
    output_jsonl: str
    jpeg_quality: int
    demo_findings: bool
    dashboard: DashboardConfig
    cameras: List[CameraConfig]
    auto_capture: bool = False
    auto_capture_interval: float = 10.0
    auto_capture_max: int = 500
    detection_enabled: bool = True
    inference_cooldown: float = 0.25  # seconds to rest between AI runs (keeps video smooth).
                                      # LATENCY KNOB: set 0 on GPU for the lowest detection latency.
    display_jpeg_quality: int = 90    # quality for the live-view JPEG (same level as before, no
                                      # quality regression). Lower it only if you want to trade a
                                      # little image quality for faster encode/transfer/decode.
    profile: bool = False             # record per-stage latency (p50/p95/p99) at /perf.json
    # --- AI-only image processing (never shown to the operator) ---
    ai_enhance: bool = True            # boost saturation + contrast on the frame fed to YOLO
    ai_max_infer_width: int = 0        # LATENCY KNOB: cap the width of the enhance+inference frame
                                       # (0 = full res). e.g. 960 cuts enhancement cost a lot; the
                                       # operator video stays full-res and boxes are scaled back.
    color_buoy_radar: bool = False     # OPT-IN: flag vivid colour blobs on the water as buoys. Great
                                       # for a lone buoy in open water, but floods false "seamarks" in
                                       # a marina (colourful boats/fenders), so OFF by default.
    color_buoy_threshold: int = 12     # min local colour-contrast (after top-hat) to count as a candidate
    color_buoy_max: int = 6            # keep only the N strongest candidates per frame (caps clutter)
    # --- Multi-object tracking with confidence hysteresis (stable live detection) ---
    track_start_conf: float = 0.60     # hysteresis HIGH: confidence needed to START a new track
    track_keep_conf: float = 0.35      # hysteresis LOW: confidence needed to SUSTAIN an existing track
    track_confirm_hits: int = 3        # must be seen in N of the last `track_confirm_window` passes to alarm
    track_confirm_window: int = 5
    track_buffer: int = 15             # keep coasting a lost track this many passes (tolerate occlusion)
    track_match_dist: float = 0.06     # max centre move to match a track (fraction of frame width)


def config_from_dict(data: Dict[str, Any]) -> AppConfig:
    app = data.get("app", {})
    dashboard = data.get("dashboard", {})
    cameras = [
        CameraConfig(
            camera_id=str(camera["camera_id"]).strip(),
            name=str(camera.get("name") or camera["camera_id"]).strip(),
            source=str(camera["source"]).strip(),
            enabled=bool(camera.get("enabled", True)),
            detect=bool(camera.get("detect", True)),
            allowed_classes=class_list(camera.get("allowed_classes")),
            ignore_zones=[
                {
                    "x1": float(zone.get("x1", 0.0)),
                    "y1": float(zone.get("y1", 0.0)),
                    "x2": float(zone.get("x2", 0.0)),
                    "y2": float(zone.get("y2", 0.0)),
                }
                for zone in camera.get("ignore_zones", [])
            ],
            ignore_polygons=[
                [{"x": float(pt.get("x", 0.0)), "y": float(pt.get("y", 0.0))} for pt in poly]
                for poly in (camera.get("ignore_polygons") or [])
                if isinstance(poly, list) and len(poly) >= 3
            ],
        )
        for camera in data.get("cameras", [])
        if str(camera.get("camera_id", "")).strip() and str(camera.get("source", "")).strip()
    ]
    if not cameras:
        raise ValueError("At least one camera with camera_id and source is required.")

    return AppConfig(
        host=str(app.get("host", "127.0.0.1")),
        port=int(app.get("port", 8765)),
        model=str(app.get("model", "yolo26n.pt")),
        confidence_threshold=float(app.get("confidence_threshold", 0.35)),
        iou_threshold=float(app.get("iou_threshold", 0.45)),
        image_size=int(app.get("image_size", 640)),
        device=None if app.get("device") in (None, "") else str(app.get("device")),
        output_jsonl=str(app.get("output_jsonl", "outputs/events/live_detections.jsonl")),
        jpeg_quality=int(app.get("jpeg_quality", 80)),
        demo_findings=bool(app.get("demo_findings", False)),
        auto_capture=bool(app.get("auto_capture", False)),
        auto_capture_interval=float(app.get("auto_capture_interval", 10.0)),
        auto_capture_max=int(app.get("auto_capture_max", 500)),
        detection_enabled=bool(app.get("detection_enabled", True)),
        inference_cooldown=float(app.get("inference_cooldown", 0.25)),
        display_jpeg_quality=int(app.get("display_jpeg_quality", 90)),
        profile=bool(app.get("profile", False)),
        ai_enhance=bool(app.get("ai_enhance", True)),
        ai_max_infer_width=int(app.get("ai_max_infer_width", 0)),
        color_buoy_radar=bool(app.get("color_buoy_radar", False)),
        color_buoy_threshold=int(app.get("color_buoy_threshold", 12)),
        color_buoy_max=int(app.get("color_buoy_max", 6)),
        track_start_conf=float(app.get("track_start_conf", 0.60)),
        track_keep_conf=float(app.get("track_keep_conf", 0.35)),
        track_confirm_hits=int(app.get("track_confirm_hits", 3)),
        track_confirm_window=int(app.get("track_confirm_window", 5)),
        track_buffer=int(app.get("track_buffer", 15)),
        track_match_dist=float(app.get("track_match_dist", 0.06)),
        dashboard=DashboardConfig(
            columns=str(dashboard.get("columns", "auto")),
            show_events=bool(dashboard.get("show_events", True)),
            card_min_width=int(dashboard.get("card_min_width", 420)),
            view_mode=str(dashboard.get("view_mode", "workspace")),
            show_labels=bool(dashboard.get("show_labels", True)),
            show_status=bool(dashboard.get("show_status", True)),
            show_vessel_area=bool(dashboard.get("show_vessel_area", True)),
        ),
        cameras=cameras,
    )


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    return {
        "app": {
            "host": config.host,
            "port": config.port,
            "model": config.model,
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
            "image_size": config.image_size,
            "device": config.device or "",
            "output_jsonl": config.output_jsonl,
            "jpeg_quality": config.jpeg_quality,
            "demo_findings": config.demo_findings,
            "auto_capture": config.auto_capture,
            "auto_capture_interval": config.auto_capture_interval,
            "auto_capture_max": config.auto_capture_max,
            "detection_enabled": config.detection_enabled,
            "inference_cooldown": config.inference_cooldown,
            "display_jpeg_quality": config.display_jpeg_quality,
            "profile": config.profile,
            "ai_enhance": config.ai_enhance,
            "ai_max_infer_width": config.ai_max_infer_width,
            "color_buoy_radar": config.color_buoy_radar,
            "color_buoy_threshold": config.color_buoy_threshold,
            "color_buoy_max": config.color_buoy_max,
            "track_start_conf": config.track_start_conf,
            "track_keep_conf": config.track_keep_conf,
            "track_confirm_hits": config.track_confirm_hits,
            "track_confirm_window": config.track_confirm_window,
            "track_buffer": config.track_buffer,
            "track_match_dist": config.track_match_dist,
        },
        "dashboard": {
            "columns": config.dashboard.columns,
            "show_events": config.dashboard.show_events,
            "card_min_width": config.dashboard.card_min_width,
            "view_mode": config.dashboard.view_mode,
            "show_labels": config.dashboard.show_labels,
            "show_status": config.dashboard.show_status,
            "show_vessel_area": config.dashboard.show_vessel_area,
        },
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "name": camera.name,
                "source": camera.source,
                "enabled": camera.enabled,
                "detect": camera.detect,
                "allowed_classes": camera.allowed_classes or [],
                "ignore_zones": camera.ignore_zones,
                "ignore_polygons": camera.ignore_polygons or [],
            }
            for camera in config.cameras
        ],
    }


def load_config(path: str | Path) -> AppConfig:
    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def save_config(path: str | Path, config: AppConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(config_to_dict(config), sort_keys=False), encoding="utf-8")


class PerfStats:
    """Rolling per-stage latency stats (milliseconds) with p50/p95/p99.

    Bounded ring buffer per stage so it never grows; thread-safe. Used to MEASURE
    the live pipeline rather than guess where the latency is. Exposed at /perf.json.
    """

    def __init__(self, window: int = 240) -> None:
        self.window = window
        self._data: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def add(self, stage: str, ms: float) -> None:
        with self._lock:
            dq = self._data.get(stage)
            if dq is None:
                dq = deque(maxlen=self.window)
                self._data[stage] = dq
            dq.append(ms)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        with self._lock:
            for stage, dq in self._data.items():
                if not dq:
                    continue
                xs = sorted(dq)
                n = len(xs)
                pick = lambda p: xs[min(n - 1, int(p * n))]  # noqa: E731
                out[stage] = {
                    "n": n,
                    "p50": round(pick(0.50), 1),
                    "p95": round(pick(0.95), 1),
                    "p99": round(pick(0.99), 1),
                    "max": round(xs[-1], 1),
                    "mean": round(sum(xs) / n, 1),
                }
        return out


class EventWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write_many(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        with self.lock, self.path.open("a", encoding="utf-8") as event_file:
            for event in events:
                event_file.write(json.dumps(event, separators=(",", ":")) + "\n")


class LatestFrameReader:
    def __init__(self, source: str) -> None:
        self.source = source
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame: Optional[Any] = None
        self.frame_id = 0
        self.frame_ts = 0.0  # perf_counter() when this frame was captured (for end-to-end latency)
        self.status = "starting"
        self.thread = threading.Thread(target=self.run, name="latest-frame-reader", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def snapshot(self) -> tuple[Optional[Any], int, str, float]:
        # No copy: the capture loop always binds a brand-new ndarray (PyAV/OpenCV
        # return fresh buffers) and never mutates one in place, and no consumer
        # writes to the shared frame — so returning the reference is safe and saves
        # a full-frame memcpy on every delivered frame.
        with self.lock:
            return self.frame, self.frame_id, self.status, self.frame_ts

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def run(self) -> None:
        while not self.stop_event.is_set():
            capture = open_capture(self.source)
            if not capture.isOpened():
                self.set_status("reconnecting")
                time.sleep(1)
                continue

            self.set_status("running")
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.set_status("stream lost")
                    break
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                    self.frame_ts = time.perf_counter()
                    self.status = "running"

            capture.release()
            time.sleep(0.2)


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        app: AppConfig,
        model: YOLO,
        active_model: str,
        model_lock: threading.Lock,
        event_writer: EventWriter,
        detect_flag: Optional[threading.Event] = None,
    ) -> None:
        self.camera = camera
        self.app = app
        self.detect_flag = detect_flag
        self.model = model
        self.active_model = active_model
        self.model_lock = model_lock
        self.event_writer = event_writer
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_jpeg: Optional[bytes] = None
        self.latest_raw_frame: Optional[Any] = None  # newest raw frame (encoded lazily on demand)
        self.latest_image_size: Optional[tuple[int, int]] = None
        self.latest_events: List[Dict[str, Any]] = []
        self.status = "starting"
        self.frame_index = 0
        self.last_auto_capture = 0.0
        self.infer_input: Optional[Any] = None  # newest frame for the inference thread
        self.infer_input_ts: float = 0.0        # capture time of infer_input (for end-to-end latency)
        self.perf = PerfStats()
        self._infer_iter = 0  # inference loop counter (drives profile rotation + tracker steps)
        self.tracker = ObjectTracker(
            start_conf=app.track_start_conf, keep_conf=app.track_keep_conf,
            confirm_hits=app.track_confirm_hits, confirm_window=app.track_confirm_window,
            buffer_frames=app.track_buffer, match_dist=app.track_match_dist,
        )
        self.thread = threading.Thread(target=self.run, name=f"camera-{camera.camera_id}", daemon=True)
        self.infer_thread = threading.Thread(target=self.run_inference, name=f"infer-{camera.camera_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()
        self.infer_thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float = 2.0) -> None:
        self.thread.join(timeout=timeout)
        self.infer_thread.join(timeout=timeout)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            size = self.latest_image_size
            detections = [self.detection_payload(e, size) for e in self.latest_events] if size else []
            return {
                "camera_id": self.camera.camera_id,
                "name": self.camera.name,
                "source": masked_source(self.camera.source),
                "status": self.status,
                "detect": self.camera.detect,
                "frame_index": self.frame_index,
                "events": self.latest_events[-10:],
                "detections": detections,
                "image_size": {"width": size[0], "height": size[1]} if size else None,
                "has_frame": self.latest_jpeg is not None,
            }

    def detection_payload(self, event: Dict[str, Any], size: tuple[int, int]) -> Dict[str, Any]:
        bbox = event["bbox_xyxy"]
        w, h = size
        return {
            "class_name": event["class_name"],
            "label": self.display_class_name(event["class_name"]),
            "severity": self.marker_severity(event["class_name"]),
            "confidence": round(float(event.get("confidence", 0.0)), 2),
            "cx": ((bbox["x1"] + bbox["x2"]) / 2) / w,
            "y1": bbox["y1"] / h,
        }

    @contextmanager
    def _timed(self, stage: str):
        """Record a stage's wall-clock latency when profiling is on; ~no-op otherwise."""
        if not self.app.profile:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.perf.add(stage, (time.perf_counter() - start) * 1000.0)

    def get_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_raw_snapshot(self) -> tuple[Optional[bytes], Optional[tuple[int, int]]]:
        # Encode the raw frame lazily — the editor/auto-capture need it rarely, so we
        # keep it out of the per-frame display hot path.
        with self.lock:
            frame = self.latest_raw_frame
            size = self.latest_image_size
        if frame is None:
            return None, size
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return (enc.tobytes() if ok else None), size

    def run(self) -> None:
        if is_live_source(self.camera.source):
            self.run_live_source()
        else:
            self.run_recorded_source()

    def run_live_source(self) -> None:
        reader = LatestFrameReader(self.camera.source)
        reader.start()
        last_frame_id = -1
        try:
            while not self.stop_event.is_set():
                frame, frame_id, status, frame_ts = reader.snapshot()
                self.set_status(status)
                if frame is None or frame_id == last_frame_id:
                    time.sleep(0.005)  # short poll: pick up a new frame with minimal added latency
                    continue
                last_frame_id = frame_id
                self.show_frame(frame, frame_ts)
        finally:
            reader.stop()

    def run_recorded_source(self) -> None:
        while not self.stop_event.is_set():
            capture = open_capture(self.camera.source)
            if not capture.isOpened():
                self.set_status("reconnecting")
                time.sleep(3)
                continue

            self.set_status("running")
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    # Local video files loop continuously for a live-style demo.
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                    if not ok:
                        self.set_status("stream lost")
                        break

                self.show_frame(frame, time.perf_counter())

            capture.release()
            time.sleep(1)

    def show_frame(self, frame: Any, capture_ts: float = 0.0) -> None:
        """Display path: encode the newest frame for the live view. This runs at
        camera speed and never waits for AI, so the video stays smooth. The
        detection markers are drawn client-side from `latest_events`, which the
        inference thread updates independently.

        Encodes exactly ONE JPEG per frame: the display image (with the vessel-area
        overlay only when it's shown). The clean raw frame is kept as an ndarray and
        encoded lazily in get_raw_snapshot, so the rarely-used editor/auto-capture
        path stays out of the hot loop."""
        quality = int(self.app.display_jpeg_quality)
        if (self.camera.ignore_zones or self.camera.ignore_polygons) and self.app.dashboard.show_vessel_area:
            disp = frame.copy()
            self.draw_ignore_zones(disp)
        else:
            disp = frame
        with self._timed("encode"):
            ok, encoded = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return
        with self.lock:
            self.latest_jpeg = encoded.tobytes()
            self.latest_raw_frame = frame
            self.latest_image_size = (int(frame.shape[1]), int(frame.shape[0]))
            self.infer_input = frame
            self.infer_input_ts = capture_ts
            self.frame_index += 1
            self.status = "running"
        self.maybe_auto_capture()

    def run_inference(self) -> None:
        """Inference path: continuously analyse the newest frame, at whatever
        speed the hardware allows, without blocking the live video."""
        while not self.stop_event.is_set():
            # AI is paused for this feed if either the global toggle is off OR this
            # camera's own AI toggle is off.
            global_off = self.detect_flag is not None and not self.detect_flag.is_set()
            if global_off or not self.camera.detect:
                with self.lock:
                    if self.latest_events:
                        self.latest_events = []
                if self.tracker.tracks:
                    self.tracker.tracks = []  # reset so it doesn't resume mid-track
                time.sleep(0.1)
                continue
            with self.lock:
                frame = self.infer_input
                capture_ts = self.infer_input_ts
            if frame is None:
                time.sleep(0.03)
                continue
            self._infer_iter += 1
            # Optional: shrink the frame for the (costly) enhance+inference path. The
            # operator video stays full-res; YOLO boxes are scaled back below so the
            # tracker/markers/colour-radar all stay in full-resolution coordinates.
            scale = 1.0
            src = frame
            mw = self.app.ai_max_infer_width
            if mw and frame.shape[1] > mw:
                scale = mw / float(frame.shape[1])
                src = cv2.resize(frame, (mw, max(1, int(round(frame.shape[0] * scale)))),
                                 interpolation=cv2.INTER_AREA)
            # AI-only enhancement: YOLO sees an auto-levelled, condition-tuned copy
            # (profile rotates each pass); the operator's video is the raw frame.
            with self._timed("enhance"):
                infer_frame = self.enhance_for_ai(src) if self.app.ai_enhance else src
            try:
                with self.model_lock, self._timed("inference"):
                    # Detect at the LOW (keep) threshold so the tracker can sustain
                    # faint objects; the tracker's hysteresis gates what becomes a track.
                    result = self.model.predict(
                        infer_frame,
                        conf=min(self.app.confidence_threshold, self.app.track_keep_conf),
                        iou=self.app.iou_threshold,
                        imgsz=self.app.image_size,
                        device=self.app.device,
                        verbose=False,
                    )[0]
            except Exception as exc:
                print(f"inference error ({self.camera.camera_id}): {exc}")
                time.sleep(0.5)
                continue
            # Gather this pass's raw detections from both sources, then let the tracker
            # confirm/hold/coast them (stable IDs, hysteresis, occlusion tolerance).
            with self._timed("postprocess"):
                dets = self.events_from_result(result)
                if scale != 1.0:
                    self._rescale_events(dets, 1.0 / scale, frame.shape[1], frame.shape[0])
                if self.app.color_buoy_radar:
                    dets.extend(self.color_buoy_candidates(frame, dets))
                if self.app.demo_findings:
                    dets.extend(self.demo_maritime_events(result.orig_img, len(dets)))
            with self._timed("tracking"):
                events = self.tracker.update(dets, self._infer_iter)
            self.event_writer.write_many(events)
            with self.lock:
                self.latest_events = events
            if self.app.profile and capture_ts:
                # End-to-end: camera capture -> detection result ready for the UI.
                self.perf.add("end_to_end", (time.perf_counter() - capture_ts) * 1000.0)
            # Rest between AI runs so the (separate) video pipeline always has CPU.
            # This makes detection a little slower but keeps the video smooth.
            cooldown = getattr(self.app, "inference_cooldown", 0.25)
            if cooldown > 0:
                time.sleep(cooldown)

    @staticmethod
    def _rescale_events(events: List[Dict[str, Any]], factor: float, full_w: int, full_h: int) -> None:
        """Scale detection boxes from the reduced inference frame back to full-res, so
        everything downstream (tracker, markers, colour radar) shares one coordinate space."""
        for ev in events:
            bb = ev["bbox_xyxy"]
            bb["x1"] *= factor; bb["y1"] *= factor; bb["x2"] *= factor; bb["y2"] *= factor
            bw = ev["bbox_xywh"]
            bw["x"] *= factor; bw["y"] *= factor; bw["width"] *= factor; bw["height"] *= factor
            ev["image_size"] = {"width": int(full_w), "height": int(full_h)}

    def current_enhance_profile(self) -> str:
        """The profile this inference pass uses. Rotates per pass so that across one
        cycle the AI has viewed the scene under day / low-light / anti-glare / haze-rain."""
        return ENHANCE_PROFILES[self._infer_iter % len(ENHANCE_PROFILES)]

    def enhance_for_ai(self, frame: Any) -> Any:
        """Return an AI-only enhanced copy of the frame using the current rotating
        profile (auto-level + condition-tuned boost) so no single fixed setting can
        blind the AI in sun/dusk/rain/haze. NEVER shown to the operator (the display
        path uses the raw frame); this feeds inference only."""
        try:
            return enhance_for_profile(frame, self.current_enhance_profile())
        except Exception:
            return frame  # never break inference over an enhancement hiccup

    def color_buoy_candidates(self, frame: Any, yolo_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Colour-saliency 'buoy radar': per-frame candidate blobs of vivid red/green/
        yellow on the water (where YOLO is weak on distant seamarks). Returns raw
        detection events for THIS frame only — the shared ObjectTracker applies the
        persistence/hysteresis so glint and noise don't trigger alerts. Runs on the RAW
        frame for true colour; results become markers on the clean video."""
        allowed = set(self.camera.allowed_classes or [])
        if allowed and "navigation_buoy" not in allowed:
            return []

        height, width = frame.shape[:2]
        b, g, r = cv2.split(frame.astype(np.int16))
        redness = r - np.maximum(g, b)
        greenness = g - np.maximum(r, b)
        yellowness = np.minimum(r, g) - b
        colorful = np.clip(np.maximum.reduce([redness, greenness, yellowness]), 0, 255).astype(np.uint8)
        # Top-hat keeps small LOCAL colour peaks (buoys) and removes broad texture/
        # gradients (water sheen, moiré from filming a screen) — a buoy is a dot, not a wash.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        colorful = cv2.morphologyEx(colorful, cv2.MORPH_TOPHAT, kernel)
        # Only look at the open-water band: skip sky/land (top) and own deck (bottom).
        band = np.zeros((height, width), np.uint8)
        band[int(0.28 * height):int(0.80 * height), :] = 255
        mask = cv2.bitwise_and((colorful > self.app.color_buoy_threshold).astype(np.uint8) * 255, band)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def in_yolo_box(cx: float, cy: float) -> bool:
            for ev in yolo_events:
                bb = ev["bbox_xyxy"]
                if bb["x1"] <= cx <= bb["x2"] and bb["y1"] <= cy <= bb["y2"]:
                    return True
            return False

        detections = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 200 or h > 200 or w * h < 4:  # buoys are small, not big colour washes
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            if in_yolo_box(cx, cy):  # already named by the net
                continue
            peak = int(colorful[y:y + h, x:x + w].max())
            detections.append((cx, cy, w, h, peak))
        # Keep only the strongest few candidates per frame so noise can't flood the view.
        detections.sort(key=lambda d: -d[4])
        detections = detections[: max(1, self.app.color_buoy_max)]

        events = []
        for cx, cy, w, h, peak in detections:
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(float(width), cx + w / 2.0)
            y2 = min(float(height), cy + h / 2.0)
            if self.is_ignored(x1, y1, x2, y2, width, height):
                continue
            confidence = min(0.95, 0.5 + peak / 30.0)  # colour strength → pseudo-confidence
            events.append(
                {
                    "schema_version": "boat_vision.detection.v1",
                    "event_type": "object_detection",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "camera_id": self.camera.camera_id,
                    "source": masked_source(self.camera.source),
                    "frame_index": self.frame_index,
                    "video_time_sec": None,
                    "model": f"{self.active_model}+color",
                    "class_id": 2000,
                    "class_name": "navigation_buoy",
                    "confidence": float(confidence),
                    "tracking_id": None,
                    "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "bbox_xywh": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                    "image_size": {"width": int(width), "height": int(height)},
                    "vessel_pose": None,
                    "color_radar": True,
                }
            )
        return events

    def maybe_auto_capture(self) -> None:
        """Passively save a clean frame every N seconds while running, for later labeling."""
        if not self.app.auto_capture:
            return
        now = time.time()
        if now - self.last_auto_capture < max(1.0, self.app.auto_capture_interval):
            return
        raw, _ = self.get_raw_snapshot()
        if not raw:
            return
        self.last_auto_capture = now
        out_dir = Path("data/datasets/maritime/raw_frames") / self.camera.camera_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        (out_dir / f"{self.camera.camera_id}_auto_{stamp}_{uuid.uuid4().hex[:6]}.jpg").write_bytes(raw)
        self.enforce_capture_cap(out_dir)

    def enforce_capture_cap(self, out_dir: Path) -> None:
        """Keep only the newest N auto-captured frames so a long trip can't fill the disk."""
        cap = self.app.auto_capture_max
        if cap <= 0:
            return
        files = sorted(out_dir.glob(f"{self.camera.camera_id}_auto_*.jpg"))
        for stale in files[:-cap] if len(files) > cap else []:
            try:
                stale.unlink()
            except OSError:
                break

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def draw_operator_overlay(self, frame: Any, events: List[Dict[str, Any]]) -> None:
        # Detection markers are rendered as crisp Font Awesome HTML overlays in the
        # browser (see status.json `detections`), not burned into the video frame.
        return

    def marker_severity(self, class_name: str) -> str:
        """Map a class to an OpenBridge-style alert tier: alarm / info / neutral."""
        if class_name in {"person", "floating_object"}:
            return "alarm"
        if class_name in {
            "navigation_buoy",
            "red_lateral_mark",
            "green_lateral_mark",
            "cardinal_mark",
            "special_mark",
        }:
            return "info"
        return "neutral"

    def draw_marker(self, frame: Any, x1: int, y1: int, x2: int, y2: int, label: str, class_name: str, confidence: float) -> None:
        """Draw a floating circular AR marker over a detection (Kystdata/OpenBridge style)."""
        severity = self.marker_severity(class_name)
        # BGR fills + glyphs per tier.
        fill = {"alarm": (54, 67, 224), "info": (110, 178, 60), "neutral": (240, 240, 240)}[severity]
        glyph = {"alarm": "!", "info": "i", "neutral": ""}[severity]
        glyph_color = (255, 255, 255) if severity != "neutral" else (40, 48, 56)

        h, w = frame.shape[:2]
        cx = max(0, min(w - 1, (x1 + x2) // 2))
        cy = max(0, min(h - 1, y1))  # pin above the object, like the reference
        radius = max(13, min(26, int(min(w, h) / 48)))

        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius + 3, (20, 26, 32), -1, cv2.LINE_AA)  # subtle dark halo
        cv2.circle(overlay, (cx, cy), radius, fill, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)  # white ring

        if glyph:
            gscale = radius / 18.0
            gsize, gbase = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_DUPLEX, gscale, 2)
            cv2.putText(
                frame, glyph,
                (cx - gsize[0] // 2, cy + gsize[1] // 2),
                cv2.FONT_HERSHEY_DUPLEX, gscale, glyph_color, 2, cv2.LINE_AA,
            )

        # compact label chip under the marker
        text = f"{label} {confidence:.2f}" if confidence else label
        font = cv2.FONT_HERSHEY_SIMPLEX
        tscale = max(0.42, min(w, h) / 1700)
        tthick = max(1, int(tscale * 2.2))
        tsize, tbase = cv2.getTextSize(text, font, tscale, tthick)
        pad = 6
        bx1 = cx - tsize[0] // 2 - pad
        by1 = cy + radius + 6
        bx2 = cx + tsize[0] // 2 + pad
        by2 = by1 + tsize[1] + tbase + pad
        bx1 = max(0, bx1); bx2 = min(w - 1, bx2)
        chip = frame.copy()
        cv2.rectangle(chip, (bx1, by1), (bx2, by2), (24, 30, 36), -1, cv2.LINE_AA)
        cv2.addWeighted(chip, 0.6, frame, 0.4, 0, frame)
        cv2.putText(
            frame, text, (bx1 + pad, by2 - tbase - 2),
            font, tscale, (255, 255, 255), tthick, cv2.LINE_AA,
        )

    def display_class_name(self, class_name: str) -> str:
        if class_name in {
            "navigation_buoy",
            "red_lateral_mark",
            "green_lateral_mark",
            "cardinal_mark",
            "special_mark",
        }:
            return "Seamark"
        names = {
            "boat": "Boat",
            "ship": "Ship",
            "sailboat": "Sailboat",
            "small_vessel": "Small vessel",
            "kayak": "Kayak",
            "floating_object": "Object",
            "person": "Person",
        }
        return names.get(class_name, class_name.replace("_", " ").title())

    def class_color(self, class_name: str) -> tuple[int, int, int]:
        if class_name in {"navigation_buoy", "cardinal_mark", "special_mark"}:
            return (210, 210, 0)
        if class_name == "red_lateral_mark":
            return (35, 35, 235)
        if class_name == "green_lateral_mark":
            return (80, 170, 45)
        if class_name in {"boat", "ship", "sailboat", "small_vessel"}:
            return (105, 125, 255)
        if class_name == "person":
            return (230, 150, 55)
        return (230, 230, 230)

    def draw_detection_box(self, frame: Any, x1: int, y1: int, x2: int, y2: int, label: str, class_name: str) -> None:
        color = self.class_color(class_name)
        thickness = max(2, int(min(frame.shape[:2]) / 360))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.65, min(frame.shape[1], frame.shape[0]) / 900)
        label_thickness = max(2, thickness)
        text_size, baseline = cv2.getTextSize(label, font, scale, label_thickness)
        pad_x = 8
        pad_y = 7
        label_x1 = max(0, x1)
        label_y2 = max(text_size[1] + pad_y * 2, y1)
        label_y1 = max(0, label_y2 - text_size[1] - baseline - pad_y * 2)
        label_x2 = min(frame.shape[1] - 1, label_x1 + text_size[0] + pad_x * 2)

        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (label_x1 + pad_x, label_y2 - baseline - pad_y),
            font,
            scale,
            (255, 255, 255),
            label_thickness,
            cv2.LINE_AA,
        )

    def draw_brackets(self, frame: Any, x1: int, y1: int, x2: int, y2: int) -> None:
        color = (255, 255, 255)
        length = max(14, min(34, int(min(x2 - x1, y2 - y1) * 0.18)))
        thickness = 2
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    def draw_ignore_zones(self, frame: Any) -> None:
        import numpy as np  # noqa: PLC0415
        height, width = frame.shape[:2]
        overlay = frame.copy()
        drew = False
        for zone in (self.camera.ignore_zones or []):
            x1 = int(zone["x1"] * width); y1 = int(zone["y1"] * height)
            x2 = int(zone["x2"] * width); y2 = int(zone["y2"] * height)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 2)
            drew = True
        for poly in (self.camera.ignore_polygons or []):
            pts = np.array([[int(p["x"] * width), int(p["y"] * height)] for p in poly], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], (80, 80, 80))
                cv2.polylines(frame, [pts], True, (60, 60, 60), 2)
                drew = True
        if drew:
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    @staticmethod
    def _point_in_polygon(px: float, py: float, poly: List[Dict[str, float]]) -> bool:
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]["x"], poly[i]["y"]
            xj, yj = poly[j]["x"], poly[j]["y"]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def is_ignored(self, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> bool:
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        for zone in (self.camera.ignore_zones or []):
            if zone["x1"] <= center_x <= zone["x2"] and zone["y1"] <= center_y <= zone["y2"]:
                return True
        for poly in (self.camera.ignore_polygons or []):
            if len(poly) >= 3 and self._point_in_polygon(center_x, center_y, poly):
                return True
        return False

    def events_from_result(self, result: Any) -> List[Dict[str, Any]]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        height, width = result.orig_shape
        allowed = set(self.camera.allowed_classes or [])
        events = []
        for xyxy, confidence, class_value in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        ):
            class_id = int(class_value)
            class_name = normalize_class(str((result.names or {}).get(class_id, class_id)))
            if allowed and class_name not in allowed:
                continue

            x1, y1, x2, y2 = [float(value) for value in xyxy]
            if self.is_ignored(x1, y1, x2, y2, int(width), int(height)):
                continue
            events.append(
                {
                    "schema_version": "boat_vision.detection.v1",
                    "event_type": "object_detection",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "camera_id": self.camera.camera_id,
                    "source": masked_source(self.camera.source),
                    "frame_index": self.frame_index,
                    "video_time_sec": None,
                    "model": self.active_model,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "tracking_id": None,
                    "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "bbox_xywh": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                    "image_size": {"width": int(width), "height": int(height)},
                    "vessel_pose": None,
                }
            )
        return events

    def demo_maritime_events(self, frame: Any, start_index: int) -> List[Dict[str, Any]]:
        height, width = frame.shape[:2]
        frame_phase = (self.frame_index % 160) / 160
        specs = [
            ("navigation_buoy", 0.91, 0.18 + frame_phase * 0.04, 0.42, 0.035, 0.07),
            ("red_lateral_mark", 0.86, 0.58, 0.36 + frame_phase * 0.03, 0.035, 0.08),
            ("green_lateral_mark", 0.83, 0.74, 0.45, 0.035, 0.08),
            ("cardinal_mark", 0.78, 0.49, 0.29, 0.032, 0.075),
        ]
        events = []
        for offset, (class_name, confidence, cx, cy, bw, bh) in enumerate(specs):
            x1 = max(0.0, (cx - bw / 2) * width)
            y1 = max(0.0, (cy - bh / 2) * height)
            x2 = min(float(width), (cx + bw / 2) * width)
            y2 = min(float(height), (cy + bh / 2) * height)
            if self.is_ignored(x1, y1, x2, y2, width, height):
                continue
            events.append(
                {
                    "schema_version": "boat_vision.detection.v1",
                    "event_type": "object_detection",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "camera_id": self.camera.camera_id,
                    "source": masked_source(self.camera.source),
                    "frame_index": self.frame_index,
                    "video_time_sec": None,
                    "model": f"{self.active_model}+demo",
                    "class_id": 1000 + start_index + offset,
                    "class_name": class_name,
                    "confidence": confidence,
                    "tracking_id": None,
                    "bbox_xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "bbox_xywh": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                    "image_size": {"width": int(width), "height": int(height)},
                    "vessel_pose": None,
                    "demo": True,
                }
            )
        return events


class DashboardState:
    def __init__(self, config_path: str | Path, config: AppConfig) -> None:
        self.config_path = Path(config_path)
        self.lock = threading.Lock()
        config.device = resolve_device(config.device)
        self.config = config
        self.model, self.active_model, self.model_fallback = load_yolo_model(config.model)
        self.model_lock = threading.Lock()
        self.event_writer = EventWriter(config.output_jsonl)
        self.detect_flag = threading.Event()
        if config.detection_enabled:
            self.detect_flag.set()
        self.workers: Dict[str, CameraWorker] = {}
        self.start_workers()

    def set_detection(self, enabled: bool) -> None:
        with self.lock:
            if enabled:
                self.detect_flag.set()
            else:
                self.detect_flag.clear()
            self.config.detection_enabled = enabled
            save_config(self.config_path, self.config)
        return enabled

    def set_camera_detection(self, camera_id: str, enabled: bool) -> Optional[bool]:
        """Toggle AI for a single feed (independent of the global toggle)."""
        with self.lock:
            found = False
            for camera in self.config.cameras:
                if camera.camera_id == camera_id:
                    camera.detect = enabled
                    found = True
            if not found:
                return None
            worker = self.workers.get(camera_id)
            if worker is not None:
                worker.camera.detect = enabled  # live effect without a restart
            save_config(self.config_path, self.config)
        return enabled

    def start_workers(self) -> None:
        self.workers = {
            camera.camera_id: CameraWorker(
                camera,
                self.config,
                self.model,
                self.active_model,
                self.model_lock,
                self.event_writer,
                self.detect_flag,
            )
            for camera in self.config.cameras
            if camera.enabled
        }
        for worker in self.workers.values():
            worker.start()

    def stop_workers(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.join()
        self.workers = {}

    def stop(self) -> None:
        with self.lock:
            self.stop_workers()

    def reconfigure(self, config: AppConfig) -> None:
        with self.lock:
            self.stop_workers()
            config.device = resolve_device(config.device)
            self.config = config
            self.model, self.active_model, self.model_fallback = load_yolo_model(config.model)
            self.model_lock = threading.Lock()
            self.event_writer = EventWriter(config.output_jsonl)
            if config.detection_enabled:
                self.detect_flag.set()
            else:
                self.detect_flag.clear()
            save_config(self.config_path, config)
            self.start_workers()

    def perf_payload(self) -> Dict[str, Any]:
        """Per-camera latency stats (ms, p50/p95/p99). Empty unless `profile: true`."""
        with self.lock:
            return {
                "profiling": self.config.profile,
                "cameras": {cid: w.perf.summary() for cid, w in self.workers.items()},
            }

    def status_payload(self) -> Dict[str, Any]:
        with self.lock:
            active = [worker.snapshot() for worker in self.workers.values()]
            active_ids = {camera["camera_id"] for camera in active}
            inactive = [
                {
                    "camera_id": camera.camera_id,
                    "name": camera.name,
                    "source": masked_source(camera.source),
                    "status": "disabled",
                    "detect": camera.detect,
                    "frame_index": 0,
                    "events": [],
                    "has_frame": False,
                }
                for camera in self.config.cameras
                if camera.camera_id not in active_ids
            ]
            return {
                "model": self.active_model,
                "configured_model": self.config.model,
                "model_fallback": self.model_fallback,
                "detection_enabled": self.detect_flag.is_set(),
                "event_file": self.config.output_jsonl,
                "dashboard": {
                    "columns": self.config.dashboard.columns,
                    "show_events": self.config.dashboard.show_events,
                    "card_min_width": self.config.dashboard.card_min_width,
                    "view_mode": self.config.dashboard.view_mode,
                },
                "cameras": active + inactive,
            }


STATE: Optional[DashboardState] = None


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html()
        elif parsed.path == "/status.json":
            self.send_json(self.state().status_payload())
        elif parsed.path == "/config.json":
            self.send_json(config_to_dict(self.state().config))
        elif parsed.path == "/perf.json":
            self.send_json(self.state().perf_payload())
        elif parsed.path == "/stream.mjpg":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            self.send_mjpeg(camera_id)
        elif parsed.path == "/snapshot.jpg":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            self.send_snapshot(camera_id)
        elif parsed.path == "/raw_snapshot.jpg":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            self.send_raw_snapshot(camera_id)
        elif parsed.path.startswith("/static/"):
            self.send_static(parsed.path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def send_static(self, path: str) -> None:
        """Serve bundled assets (fonts/icons) so the UI works fully offline."""
        static_root = (Path(__file__).resolve().parent / "static").resolve()
        rel = path[len("/static/"):]
        target = (static_root / rel).resolve()
        # Prevent path traversal outside the static directory.
        if static_root not in target.parents and target != static_root:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        types = {
            ".css": "text/css", ".woff2": "font/woff2", ".woff": "font/woff",
            ".ttf": "font/ttf", ".js": "application/javascript", ".png": "image/png",
            ".svg": "image/svg+xml", ".ico": "image/x-icon", ".jpg": "image/jpeg",
            ".mp3": "audio/mpeg",
        }
        ctype = types.get(target.suffix.lower(), "application/octet-stream")
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/annotation.json":
            self.save_annotation()
            return
        if parsed.path == "/capture.json":
            self.capture_frames()
            return
        if parsed.path == "/detection.json":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                enabled = bool(payload.get("enabled", True))
                camera_id = payload.get("camera_id")
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if camera_id:  # per-feed toggle
                result = self.state().set_camera_detection(str(camera_id), enabled)
                if result is None:
                    self.send_json({"ok": False, "error": "unknown camera_id"}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json({"ok": True, "camera_id": camera_id, "detect": enabled})
                return
            self.state().set_detection(enabled)
            self.send_json({"ok": True, "detection_enabled": enabled})
            return
        if parsed.path != "/config.json":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            config = config_from_dict(json.loads(raw.decode("utf-8")))
            self.state().reconfigure(config)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_json({"ok": True})

    def state(self) -> DashboardState:
        assert STATE is not None
        return STATE

    def send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_html(self) -> None:
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_mjpeg(self, camera_id: str) -> None:
        worker = self.state().workers.get(camera_id)
        if worker is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            frame = worker.get_jpeg()
            if frame is None:
                time.sleep(0.2)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                break

    def send_snapshot(self, camera_id: str) -> None:
        worker = self.state().workers.get(camera_id)
        if worker is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        deadline = time.monotonic() + 3
        frame = worker.get_jpeg()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.1)
            frame = worker.get_jpeg()

        if frame is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame available yet")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def send_raw_snapshot(self, camera_id: str) -> None:
        worker = self.state().workers.get(camera_id)
        if worker is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        deadline = time.monotonic() + 3
        frame, _ = worker.get_raw_snapshot()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.1)
            frame, _ = worker.get_raw_snapshot()

        if frame is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No raw frame available yet")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def capture_frames(self) -> None:
        """Save the current clean (un-annotated) frame(s) for later labeling/training."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            payload = {}
        requested = str(payload.get("camera_id", "")).strip()
        workers = self.state().workers
        targets = [requested] if requested and requested in workers else list(workers.keys())

        base = Path("data/datasets/maritime/raw_frames")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        saved = []
        for cam_id in targets:
            worker = workers.get(cam_id)
            if worker is None:
                continue
            raw, _ = worker.get_raw_snapshot()
            if not raw:
                continue
            out_dir = base / cam_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{cam_id}_{stamp}_{uuid.uuid4().hex[:6]}.jpg"
            out_path.write_bytes(raw)
            saved.append(str(out_path))

        self.send_json({"ok": bool(saved), "count": len(saved), "saved": saved})

    def save_annotation(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            camera_id = str(payload["camera_id"])
            class_name = str(payload["class_name"])
            bbox = payload["bbox"]
            class_id = MARITIME_CLASSES.index(class_name)
        except Exception as exc:
            self.send_json({"ok": False, "error": f"Invalid annotation payload: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        worker = self.state().workers.get(camera_id)
        if worker is None:
            self.send_json({"ok": False, "error": "Camera is not running."}, status=HTTPStatus.BAD_REQUEST)
            return

        frame, image_size = worker.get_raw_snapshot()
        if frame is None or image_size is None:
            self.send_json({"ok": False, "error": "No camera frame available yet."}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        x1 = max(0.0, min(1.0, float(bbox["x1"])))
        y1 = max(0.0, min(1.0, float(bbox["y1"])))
        x2 = max(0.0, min(1.0, float(bbox["x2"])))
        y2 = max(0.0, min(1.0, float(bbox["y2"])))
        if x2 <= x1 or y2 <= y1:
            self.send_json({"ok": False, "error": "Annotation box must have positive width and height."}, status=HTTPStatus.BAD_REQUEST)
            return

        stem = f"{camera_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        image_dir = Path("data/datasets/maritime/images/train")
        label_dir = Path("data/datasets/maritime/labels/train")
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"

        image_path.write_bytes(frame)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        label_path.write_text(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n", encoding="utf-8")

        self.send_json(
            {
                "ok": True,
                "image": str(image_path),
                "label": str(label_path),
                "class_id": class_id,
                "class_name": class_name,
                "image_size": {"width": image_size[0], "height": image_size[1]},
            }
        )


DASHBOARD_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Boat Vision</title>
  <link rel="icon" type="image/png" href="/static/app_icon.png">
  <link rel="apple-touch-icon" href="/static/app_icon.png">
  <link rel="stylesheet" href="/static/inter/inter.css">
  <link rel="stylesheet" href="/static/fontawesome/css/all.min.css">
  <style>
    :root {
      --bg: #eef2f6;
      --sidebar: #0f2733;
      --sidebar-text: #cde3ee;
      --sidebar-muted: #7fa3b4;
      --panel: #ffffff;
      --panel-2: #f7f9fb;
      --line: #e3e8ee;
      --line-strong: #d3dbe3;
      --text: #16242e;
      --muted: #64748b;
      --accent: #1f9ed1;
      --accent-strong: #1683b3;
      --accent-weak: #e6f4fb;
      --danger: #c0392b;
      --danger-weak: #fdeceb;
      --ok: #0f8a5f;
      --ok-weak: #e6f6ee;
      --warn: #b97309;
      --warn-weak: #fbf1de;
      --radius: 12px;
      --radius-sm: 9px;
      --shadow: 0 1px 2px rgba(15,38,51,0.05), 0 1px 1px rgba(15,38,51,0.03);
      --shadow-md: 0 6px 18px rgba(15,38,51,0.08);
      --ring: 0 0 0 3px rgba(31,158,209,0.18);
      --t: 140ms cubic-bezier(.4,0,.2,1);
    }
    /* OpenBridge-inspired night theme: dark, low-luminance, glare-reduced for
       bridge use at night. Switched via the Day/Night control in the header. */
    html[data-theme="night"] {
      --bg: #05080b;
      --sidebar: #02060a;
      --sidebar-text: #9fb6c4;
      --sidebar-muted: #5d7585;
      --panel: #0e151b;
      --panel-2: #0a1116;
      --line: #1c2730;
      --line-strong: #2a3742;
      --text: #c7d6df;
      --muted: #738796;
      --accent: #2f9fd6;
      --accent-strong: #4bb4e6;
      --accent-weak: #0f2531;
      --danger: #e8554a;
      --danger-weak: #2a0f0e;
      --ok: #2fb27e;
      --ok-weak: #0c241a;
      --warn: #f0a020;
      --warn-weak: #2a1f08;
      --shadow: 0 1px 2px rgba(0,0,0,0.45);
      --shadow-md: 0 8px 22px rgba(0,0,0,0.55);
      --ring: 0 0 0 3px rgba(47,159,214,0.3);
    }
    /* Dusk: intermediate dimmed theme between day and night. */
    html[data-theme="dusk"] {
      --bg: #1a232c;
      --sidebar: #0c151c;
      --sidebar-text: #b4c8d4;
      --sidebar-muted: #6f8694;
      --panel: #222e38;
      --panel-2: #1c2832;
      --line: #313f4a;
      --line-strong: #3d4d59;
      --text: #d9e4ec;
      --muted: #8ba0ad;
      --accent: #2f9fd6;
      --accent-strong: #4bb4e6;
      --accent-weak: #16313f;
      --danger: #e8675c;
      --danger-weak: #341916;
      --ok: #34b583;
      --ok-weak: #133327;
      --warn: #f0a83a;
      --warn-weak: #2e2410;
      --shadow: 0 1px 2px rgba(0,0,0,0.3);
      --shadow-md: 0 8px 22px rgba(0,0,0,0.4);
    }
    * { box-sizing: border-box; }
    html { transition: background 200ms ease; }
    /* OpenBridge "brilliance" display dimming, applied to the whole UI. */
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.4; -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; filter: brightness(var(--brilliance, 1)); transition: filter 160ms ease; }
    html[data-theme="night"] img, html[data-theme="dusk"] img { filter: brightness(0.84) saturate(0.95); }
    ::selection { background: rgba(31,158,209,0.22); }
    .app { min-height: 100vh; display: grid; grid-template-columns: 264px minmax(0, 1fr) 400px; }
    .sidebar { background: var(--sidebar); color: var(--sidebar-text); display: flex; flex-direction: column; min-height: 100vh; }
    .brand { height: 72px; display: flex; align-items: center; gap: 12px; padding: 0 18px; border-bottom: 1px solid rgba(255,255,255,0.07); }
    .brand-mark { width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(150deg, var(--accent), var(--accent-strong)); display: grid; place-items: center; color: #fff; font-weight: 800; font-size: 18px; box-shadow: 0 2px 8px rgba(31,158,209,0.35); }
    .brand strong { display: block; font-size: 15px; line-height: 1.15; color: #fff; letter-spacing: -0.01em; }
    .brand span { display: block; color: var(--sidebar-muted); font-size: 12px; margin-top: 2px; }
    .nav { padding: 12px 10px; display: grid; gap: 3px; }
    .nav button { justify-content: flex-start; text-align: left; border-radius: 9px; border-color: transparent; background: transparent; color: var(--sidebar-text); font-weight: 600; min-height: 40px; transition: background var(--t), color var(--t); box-shadow: none; }
    .nav button:hover { background: rgba(255,255,255,0.06); border-color: transparent; }
    .nav button.active { background: var(--accent); border-color: var(--accent); color: #fff; box-shadow: 0 2px 8px rgba(31,158,209,0.3); }
    .sidebar-foot { margin-top: auto; padding: 16px 18px; color: var(--sidebar-muted); font-size: 12px; display: grid; gap: 9px; }
    .workspace { border-top: 1px solid rgba(255,255,255,0.08); padding-top: 13px; color: #fff; font-weight: 700; letter-spacing: 0.02em; }
    .content { min-width: 0; display: flex; flex-direction: column; }
    header { height: 72px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; background: var(--panel); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 5; }
    h1 { margin: 0; font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }
    .crumb { display: flex; align-items: center; gap: 9px; color: var(--text); font-weight: 650; }
    .crumb h1::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 999px; background: var(--ok); margin-right: 10px; box-shadow: 0 0 0 3px var(--ok-weak); vertical-align: middle; }
    .avatar { width: 38px; height: 38px; border-radius: 999px; background: linear-gradient(150deg,#34d399,#0f8a5f); color: #fff; display: grid; place-items: center; font-weight: 700; font-size: 13px; }
    .header-actions { display: flex; align-items: center; gap: 8px; }
    button, input, select, textarea { font: inherit; }
    button { border: 1px solid var(--line-strong); background: #fff; color: var(--text); min-height: 40px; padding: 0 15px; cursor: pointer; border-radius: var(--radius-sm); display: inline-flex; align-items: center; justify-content: center; gap: 7px; font-weight: 600; font-size: 13px; transition: background var(--t), border-color var(--t), box-shadow var(--t), transform var(--t); }
    button:hover { border-color: #b7c2cd; background: var(--panel-2); }
    button:active { transform: translateY(1px); }
    button:focus-visible { outline: none; box-shadow: var(--ring); }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; box-shadow: 0 1px 3px rgba(31,158,209,0.3); }
    button.primary:hover { background: var(--accent-strong); border-color: var(--accent-strong); }
    button.danger { color: var(--danger); border-color: #f1c6c2; background: var(--danger-weak); }
    button.danger:hover { background: #fbdedb; border-color: #e7a59e; }
    .video-area { min-width: 0; }
    .toolbar { min-height: 56px; display: flex; gap: 8px; align-items: center; padding: 16px 24px 14px; }
    .toolbar .seg { display: inline-flex; border: 1px solid var(--line-strong); border-radius: var(--radius-sm); overflow: hidden; background: #fff; }
    .toolbar .seg button { border: 0; border-radius: 0; min-height: 36px; box-shadow: none; border-right: 1px solid var(--line); }
    .toolbar .seg button:last-child { border-right: 0; }
    .toolbar .seg button.active { background: var(--accent-weak); color: var(--accent-strong); }
    .toolbar span { color: var(--muted); font-size: 12px; margin-left: auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-variant-numeric: tabular-nums; }
    main { display: grid; gap: 16px; padding: 0 24px 24px; }
    .toolbar-label { color: var(--muted); font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.04em; }
    .camera { border: 1px solid var(--line); background: var(--panel); min-width: 0; border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow); transition: box-shadow var(--t), border-color var(--t); }
    .camera:hover { box-shadow: var(--shadow-md); border-color: var(--line-strong); }
    .camera-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 11px 14px; border-bottom: 1px solid var(--line); }
    h2 { margin: 0; font-size: 14px; font-weight: 700; letter-spacing: -0.01em; }
    .status { color: var(--ok); background: var(--ok-weak); border: 1px solid #bfe6d3; border-radius: 999px; padding: 4px 10px 4px 8px; font-size: 12px; font-weight: 650; white-space: nowrap; display: inline-flex; align-items: center; gap: 6px; font-variant-numeric: tabular-nums; }
    .status::before { content: ''; width: 7px; height: 7px; border-radius: 999px; background: currentColor; flex: none; }
    .status.disabled { color: var(--muted); background: #eef1f4; border-color: #e0e5ea; }
    .status.error { color: var(--danger); background: var(--danger-weak); border-color: #f3c9c4; }
    img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #0c1620; }
    pre { margin: 0; padding: 11px 14px; min-height: 70px; max-height: 150px; overflow: auto; white-space: pre-wrap; font-size: 12px; color: #475569; background: var(--panel-2); border-top: 1px solid var(--line); font-variant-numeric: tabular-nums; }
    aside { border-left: 1px solid var(--line); background: var(--panel); overflow: auto; max-height: 100vh; }
    .settings-head { height: 72px; display: flex; justify-content: space-between; align-items: center; padding: 0 18px; border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--panel); z-index: 2; }
    .settings-head strong { font-size: 15px; letter-spacing: -0.01em; }
    .settings-body { padding: 16px; display: grid; gap: 16px; }
    fieldset { border: 1px solid var(--line); padding: 14px; margin: 0; display: grid; gap: 12px; border-radius: var(--radius); background: var(--panel); }
    legend { padding: 2px 8px; color: var(--accent-strong); background: var(--accent-weak); border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 12px; font-weight: 600; }
    input, select, textarea { width: 100%; border: 1px solid var(--line-strong); background: #fff; color: var(--text); min-height: 38px; padding: 8px 10px; border-radius: var(--radius-sm); font-weight: 500; transition: border-color var(--t), box-shadow var(--t); }
    input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); box-shadow: var(--ring); }
    input::placeholder, textarea::placeholder { color: #9aa7b4; }
    textarea { min-height: 72px; resize: vertical; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .camera-config { display: grid; gap: 11px; padding: 13px; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--radius); margin-bottom: 11px; }
    .camera-config-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .check { display: flex; grid-template-columns: none; flex-direction: row; align-items: center; gap: 8px; color: var(--text); }
    .check input { width: auto; min-height: auto; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .message { color: var(--muted); min-height: 18px; font-size: 12px; }
    .modal { position: fixed; inset: 0; z-index: 20; display: none; align-items: center; justify-content: center; background: rgba(15,23,42,0.52); padding: 24px; }
    .modal.open { display: flex; }
    .modal-panel { width: min(1100px, 96vw); max-height: 94vh; overflow: auto; background: #fff; border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 20px 50px rgba(15,23,42,0.24); }
    .modal-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--line); }
    .modal-head strong { font-size: 15px; }
    .modal-body { padding: 16px; display: grid; gap: 12px; }
    .zone-canvas { position: relative; background: #0c1620; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; cursor: crosshair; line-height: 0; }
    .zone-canvas img { width: 100%; height: auto; display: block; user-select: none; pointer-events: none; }
    #zone-svg { position: absolute; inset: 0; width: 100%; height: 100%; }
    #zone-svg polygon { fill: rgba(245,158,11,0.22); stroke: #f59e0b; stroke-width: 0.4; }
    #zone-svg circle { fill: #fff; stroke: #f59e0b; stroke-width: 0.4; }
    .zone-box { position: absolute; border: 2px solid #f59e0b; background: rgba(245,158,11,0.22); display: none; pointer-events: none; }
    .label-box { border-color: #0ea5e9; background: rgba(14,165,233,0.18); }
    .hint { color: var(--muted); font-size: 12px; }
    .app.settings-hidden { grid-template-columns: 276px minmax(0, 1fr); }
    .app.settings-hidden #settings { display: none; }
    .app.wall { grid-template-columns: minmax(0, 1fr); background: #101820; }
    .app.wall .sidebar,
    .app.wall #settings,
    .app.wall .toolbar,
    .app.wall .avatar { display: none; }
    .app.wall header { height: 42px; padding: 0 10px; border-bottom: 1px solid #cfd6dd; background: #fff; }
    .app.wall h1 { font-size: 15px; }
    .app.wall .header-actions { gap: 6px; }
    .app.wall button { min-height: 32px; padding: 0 10px; }
    .app.wall main { height: calc(100vh - 42px); padding: 6px; gap: 6px; grid-template-columns: repeat(2, minmax(0, 1fr)) !important; grid-template-rows: repeat(2, minmax(0, 1fr)); overflow: hidden; }
    .app.wall .camera { position: relative; display: flex; flex-direction: column; min-height: 0; border-radius: 4px; border-color: #c8d0d9; box-shadow: none; background: #0b1117; }
    .app.wall .camera-header { position: absolute; z-index: 2; top: 8px; left: 8px; right: 8px; min-height: 0; padding: 0; border: 0; pointer-events: none; }
    .app.wall .camera-header h2 { background: rgba(255,255,255,0.92); border: 1px solid rgba(209,216,224,0.95); border-radius: 999px; padding: 5px 9px; font-size: 13px; box-shadow: var(--shadow); }
    .app.wall .status { background: rgba(232,246,251,0.95); box-shadow: var(--shadow); }
    .app.wall img { flex: 1; width: 100%; height: 100%; min-height: 0; aspect-ratio: auto; object-fit: cover; }
    .app.wall pre.events { display: none !important; }
    .app.wall.focused main { padding: 0; gap: 0; grid-template-columns: minmax(0, 1fr) !important; grid-template-rows: minmax(0, 1fr); }
    .app.wall.focused .camera { display: none; border: 0; border-radius: 0; }
    .app.wall.focused .camera.focused { display: flex; }
    .app.wall.focused header { position: fixed; z-index: 4; top: 12px; left: 12px; right: 12px; height: 40px; border: 1px solid rgba(209,216,224,0.72); border-radius: 10px; background: rgba(255,255,255,0.78); backdrop-filter: blur(12px); }
    .app.wall.focused .camera.focused .camera-header { top: 64px; left: 16px; right: 16px; }
    @media (max-width: 1180px) {
      .app { grid-template-columns: 236px minmax(0, 1fr); }
      aside { grid-column: 1 / -1; border-left: 0; border-top: 1px solid var(--line); max-height: none; }
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { display: none; }
      header, .toolbar, main { padding-left: 14px; padding-right: 14px; }
    }

    /* ---- OpenBridge-style Brilliance + Day/Dusk/Night control ---- */
    .display-btn { display: inline-flex; align-items: center; gap: 8px; }
    .display-pop { position: fixed; z-index: 40; top: 66px; right: 18px; width: 268px; background: var(--panel);
      border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--shadow-md); padding: 16px; display: none;
      backdrop-filter: blur(14px); }
    .display-pop.open { display: grid; gap: 16px; }
    .display-pop h4 { margin: 0; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); display: flex; align-items: center; gap: 8px; }
    .brilliance-row { display: flex; align-items: center; gap: 12px; }
    .brilliance-row .val { margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 700; font-size: 15px; }
    input[type="range"].brilliance { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 999px;
      background: linear-gradient(90deg, var(--accent) var(--fill,60%), var(--line-strong) var(--fill,60%)); padding: 0; min-height: 0; border: 0; }
    input[type="range"].brilliance::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; border-radius: 999px;
      background: #fff; border: 1px solid var(--line-strong); box-shadow: var(--shadow-md); cursor: pointer; }
    .theme-seg { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .theme-seg button { min-height: 54px; flex-direction: column; gap: 4px; border-radius: 12px; font-size: 12px; }
    .theme-seg button .dot { width: 16px; height: 16px; border-radius: 999px; border: 1px solid rgba(0,0,0,0.15); }
    .theme-seg button[data-theme-val="day"] .dot { background: #f5b301; }
    .theme-seg button[data-theme-val="dusk"] .dot { background: #d97742; }
    .theme-seg button[data-theme-val="night"] .dot { background: #1f3a4d; }
    .theme-seg button.active { border-color: var(--accent); background: var(--accent-weak); color: var(--accent-strong); box-shadow: var(--ring); }

    /* ===== IMMERSIVE LAYOUT — full-screen camera view, no sidebars ===== */
    .app { display: block; min-height: 100vh; background: #070b0f; }
    .sidebar, .toolbar, .avatar, header { display: none !important; }
    main#camera-grid { position: fixed; inset: 0; display: grid; gap: 3px; padding: 3px;
      height: 100vh; width: 100vw; background: #070b0f; grid-auto-rows: 1fr; }
    .camera { position: relative; overflow: hidden; border-radius: 10px; background: #0b1117;
      display: flex; min-height: 0; min-width: 0; box-shadow: none; border: 0; cursor: pointer; }
    .camera:hover { box-shadow: none; border: 0; }
    .camera img { width: 100%; height: 100%; aspect-ratio: auto; object-fit: contain; background: #0b1117; }
    .overlay { position: absolute; inset: 0; z-index: 2; pointer-events: none; }
    /* Marker sits ABOVE the object; a leader line points down to it. */
    .marker { position: absolute; transform: translate(-50%, -100%); display: flex; flex-direction: column;
      align-items: center; gap: 4px; transition: left 180ms ease, top 180ms ease; }
    .marker .badge { width: 32px; height: 32px; border-radius: 999px; display: grid; place-items: center;
      color: #fff; font-size: 14px; border: 2px solid rgba(255,255,255,0.95);
      box-shadow: 0 3px 10px rgba(0,0,0,0.45); }
    .marker.alarm .badge { background: #e0443b; }
    .marker.info .badge { background: #1f9c4d; }
    .marker.neutral .badge { background: rgba(20,28,36,0.78); }
    .marker .tag { font-size: 11px; font-weight: 600; color: #0e1a24; background: rgba(255,255,255,0.86);
      border: 1px solid rgba(255,255,255,0.7); border-radius: 999px; padding: 2px 9px; white-space: nowrap;
      backdrop-filter: blur(8px); box-shadow: 0 2px 8px rgba(8,15,22,0.2); font-variant-numeric: tabular-nums; }
    .marker .leader { width: 2px; height: 28px; background: rgba(255,255,255,0.92);
      box-shadow: 0 0 3px rgba(0,0,0,0.6); }
    .marker .leader::after { content: ''; position: absolute; bottom: -3px; left: 50%; transform: translateX(-50%);
      width: 7px; height: 7px; border-radius: 999px; background: #fff; box-shadow: 0 0 3px rgba(0,0,0,0.6); }
    .app.hide-labels .camera-header h2 { display: none; }
    .app.hide-status .camera-header .status { display: none; }
    .camera-header { position: absolute; z-index: 3; top: 12px; left: 12px; right: 12px; min-height: 0;
      border: 0; padding: 0; background: transparent; display: flex; justify-content: space-between;
      align-items: flex-start; gap: 10px; pointer-events: none; }
    .camera-header h2 { color: #0e1a24; font-size: 13px; font-weight: 600; letter-spacing: -0.01em; padding: 7px 14px;
      background: rgba(255,255,255,0.82); border: 1px solid rgba(255,255,255,0.7); border-radius: 999px;
      backdrop-filter: blur(14px) saturate(1.2); box-shadow: 0 4px 16px rgba(8,15,22,0.18); }
    .camera-header .status { background: rgba(255,255,255,0.82); border: 1px solid rgba(255,255,255,0.7);
      color: #0f8a5f; font-weight: 600; backdrop-filter: blur(14px) saturate(1.2); box-shadow: 0 4px 16px rgba(8,15,22,0.18); }
    .camera-header .status.disabled { color: #64748b; }
    .camera-header .status.error { color: #c0392b; }
    .camera-header h2 { margin-right: auto; }
    .cam-ai { pointer-events: auto; cursor: pointer; font-size: 12px; font-weight: 600; color: #0e1a24;
      padding: 6px 11px; border-radius: 999px; background: rgba(255,255,255,0.82);
      border: 1px solid rgba(255,255,255,0.7); backdrop-filter: blur(14px) saturate(1.2);
      box-shadow: 0 4px 16px rgba(8,15,22,0.18); display: inline-flex; align-items: center; gap: 6px; }
    .cam-ai:hover { background: rgba(255,255,255,0.95); }
    .cam-ai .fa-brain { color: var(--ok); }
    .cam-ai.off { color: var(--danger); }
    .cam-ai.off .fa-brain { color: var(--danger); opacity: 0.85; }
    .camera pre.events, pre.events { display: none !important; }
    .app.focused #camera-grid { grid-template-columns: 1fr !important; grid-auto-rows: 1fr; }
    .app.focused .camera { display: none; }
    .app.focused .camera.focused { display: flex; }

    /* Floating top bar — light, classy glass to match the reference.
       Auto-hides: only visible when the cursor is near the top of the screen, so
       the controls stay out of the way of the live video. */
    .topbar { position: fixed; z-index: 30; top: 16px; left: 16px; right: 16px; height: 48px;
      display: flex; align-items: center; justify-content: space-between; pointer-events: none;
      transition: opacity 220ms ease, transform 260ms cubic-bezier(.4,0,.2,1); }
    .topbar > * { pointer-events: auto; }
    .topbar.hidden { opacity: 0; transform: translateY(-135%); }
    .topbar.hidden > * { pointer-events: none; }
    .tb-brand { display: flex; align-items: center; gap: 11px; padding: 7px 16px 7px 8px;
      background: rgba(255,255,255,0.82); border: 1px solid rgba(255,255,255,0.7); border-radius: 999px;
      backdrop-filter: blur(14px) saturate(1.2); box-shadow: 0 4px 16px rgba(8,15,22,0.18); color: #0e1a24; }
    .tb-mark { width: 30px; height: 30px; border-radius: 8px; object-fit: cover; display: block; }
    .tb-brand strong { font-size: 14px; font-weight: 600; letter-spacing: -0.01em; }
    .tb-actions { display: flex; gap: 10px; }
    .tb-actions .ghost { background: rgba(255,255,255,0.82); border: 1px solid rgba(255,255,255,0.7);
      color: #0e1a24; font-weight: 600; backdrop-filter: blur(14px) saturate(1.2); box-shadow: 0 4px 16px rgba(8,15,22,0.18); }
    .tb-actions .ghost:hover { background: rgba(255,255,255,0.95); }
    .tb-actions .ghost.icon { width: 44px; padding: 0; font-size: 17px; }
    #ai-toggle .fa-brain { color: var(--ok); }
    #ai-toggle.ai-off { color: var(--danger); }
    #ai-toggle.ai-off .fa-brain { color: var(--danger); opacity: 0.8; }

    /* Settings: right slide-over drawer, hidden by default */
    aside#settings { position: fixed; z-index: 60; top: 0; right: 0; height: 100vh; width: min(420px, 92vw);
      border-left: 1px solid var(--line); background: var(--panel); transform: translateX(103%);
      transition: transform 220ms cubic-bezier(.4,0,.2,1); box-shadow: -16px 0 48px rgba(0,0,0,0.4); }
    aside#settings.open { transform: translateX(0); }
    #settings-scrim { position: fixed; inset: 0; z-index: 55; background: rgba(5,8,11,0.5); opacity: 0;
      pointer-events: none; transition: opacity 200ms ease; }
    #settings-scrim.open { opacity: 1; pointer-events: auto; }

    /* Capture toast */
    .toast { position: fixed; z-index: 80; left: 50%; bottom: 28px; transform: translateX(-50%) translateY(20px);
      background: rgba(14,26,36,0.92); color: #fff; padding: 11px 18px; border-radius: 999px; font-size: 13px;
      font-weight: 600; box-shadow: 0 8px 24px rgba(0,0,0,0.4); opacity: 0; pointer-events: none;
      transition: opacity 200ms ease, transform 200ms ease; display: flex; align-items: center; gap: 9px; }
    .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
    #capture-btn.flash { background: var(--accent); border-color: var(--accent); color: #fff; }
  </style>
</head>
<body>
  <div id="app" class="app">
    <main id="camera-grid"></main>
    <div class="topbar">
      <div class="tb-brand"><img class="tb-mark" src="/static/app_icon.png" alt="Boat Vision"><strong id="focus-title">Live cameras</strong></div>
      <div class="tb-actions">
        <button id="ai-toggle" class="ghost" title="Turn the YOLO detection on/off"><i class="fa-solid fa-brain"></i> <span id="ai-label">AI on</span></button>
        <button id="capture-btn" class="ghost" title="Capture current frame(s) for later labeling"><i class="fa-solid fa-camera"></i> Capture</button>
        <button id="display-btn" class="ghost">☀ Display</button>
        <button id="toggle-settings" class="ghost icon" title="Settings"><i class="fa-solid fa-gear"></i></button>
      </div>
    </div>
    <div id="toast" class="toast"></div>
    <div id="display-pop" class="display-pop">
      <div>
        <h4>☀ Brilliance</h4>
        <div class="brilliance-row">
          <input id="brilliance" class="brilliance" type="range" min="25" max="100" step="1" value="100">
          <span class="val" id="brilliance-val">100</span>
        </div>
      </div>
      <div>
        <h4>Day / Night</h4>
        <div class="theme-seg">
          <button data-theme-val="day"><span class="dot"></span>Day</button>
          <button data-theme-val="dusk"><span class="dot"></span>Dusk</button>
          <button data-theme-val="night"><span class="dot"></span>Night</button>
        </div>
      </div>
    </div>
    <div id="settings-scrim"></div>
    <aside id="settings">
      <div class="settings-head">
        <strong>Configuration</strong>
        <div style="display:flex; gap:8px;">
          <button id="save" class="primary">Save</button>
          <button id="settings-close" class="icon" title="Close">✕</button>
        </div>
      </div>
      <div class="settings-body">
        <fieldset>
          <legend>Runtime</legend>
          <label>Model<input id="model"></label>
          <div class="row">
            <label>Device<input id="device" placeholder="mps, 0, cpu"></label>
            <label>Image size<input id="image-size" type="number" min="320" step="32"></label>
          </div>
          <div class="row">
            <label>Confidence<input id="confidence" type="number" min="0" max="1" step="0.01"></label>
            <label>JPEG quality<input id="jpeg-quality" type="number" min="30" max="95" step="1"></label>
          </div>
          <label class="check"><input id="demo-findings" type="checkbox"> Demo maritime findings</label>
          <label>Event JSONL<input id="output-jsonl"></label>
        </fieldset>

        <fieldset>
          <legend>Training capture</legend>
          <label class="check"><input id="auto-capture" type="checkbox"> Auto-capture frames for training</label>
          <div class="row">
            <label>Every (seconds)<input id="auto-capture-interval" type="number" min="1" step="1"></label>
            <label>Max frames / camera<input id="auto-capture-max" type="number" min="0" step="50"></label>
          </div>
          <div class="hint">Saves a clean frame on an interval to data/datasets/maritime/raw_frames/&lt;camera&gt;/ for later labeling. Oldest auto frames are rotated out past the cap. Takes effect after Save.</div>
        </fieldset>

        <fieldset>
          <legend>Layout</legend>
          <div class="row">
            <label>Columns
              <select id="columns">
                <option value="auto">Auto</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
              </select>
            </label>
            <label>Min card width<input id="card-min-width" type="number" min="300" step="20"></label>
          </div>
          <label class="check"><input id="show-labels" type="checkbox"> Show camera labels</label>
          <label class="check"><input id="show-status" type="checkbox"> Show status / frame counter</label>
          <label class="check"><input id="show-vessel-area" type="checkbox"> Show own-vessel area on video</label>
          <label class="check"><input id="alert-sound" type="checkbox"> Alert sound on new detections</label>
          <label class="check"><input id="show-events" type="checkbox"> Show recent events</label>
          <label class="check"><input id="start-wall-mode" type="checkbox"> Start in camera wall mode</label>
        </fieldset>

        <fieldset>
          <legend>Cameras</legend>
          <div id="camera-configs"></div>
          <div class="actions">
            <button id="add-camera">Add camera</button>
          </div>
        </fieldset>
        <div id="message" class="message"></div>
      </div>
    </aside>
    <div id="zone-modal" class="modal">
      <div class="modal-panel">
        <div class="modal-head">
          <strong>Mark own vessel area</strong>
          <button id="zone-close">Close</button>
        </div>
        <div class="modal-body">
          <div class="hint">Click around the part of the image that shows your own boat to draw a polygon (add 3+ points). Detections centered inside it are ignored.</div>
          <div id="zone-canvas" class="zone-canvas">
            <img id="zone-image" alt="Camera frame for marking own-vessel area">
            <svg id="zone-svg" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
          </div>
          <div class="actions">
            <button id="zone-undo">Undo point</button>
            <button id="zone-apply" class="primary">Use marked area</button>
            <button id="zone-clear">Clear</button>
          </div>
          <div id="zone-output" class="hint"></div>
        </div>
      </div>
    </div>
    <div id="label-modal" class="modal">
      <div class="modal-panel">
        <div class="modal-head">
          <strong>Save training label</strong>
          <button id="label-close">Close</button>
        </div>
        <div class="modal-body">
          <label>Class
            <select id="label-class"></select>
          </label>
          <div class="hint">Draw a rectangle around the object. This saves a clean image and YOLO label for later training.</div>
          <div id="label-canvas" class="zone-canvas">
            <img id="label-image" alt="Camera frame for training label">
            <div id="label-box" class="zone-box label-box"></div>
          </div>
          <div class="actions">
            <button id="label-save" class="primary">Save training label</button>
          </div>
          <div id="label-output" class="hint"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let config = null;
    let alertSoundOn = (localStorage.getItem('bv-alert-sound') !== 'off');
    const alertAudio = new Audio('/static/sounds/alert.mp3');
    let lastAlertTime = 0;
    const prevCatCounts = {};   // per camera_id -> {person, mark, vessel}
    let activeZoneTextarea = null;
    let activeZone = null;
    let dragStart = null;
    let activeLabelCameraId = null;
    let activeLabelBox = null;
    let labelDragStart = null;
    const maritimeClasses = [
      'boat',
      'ship',
      'sailboat',
      'small_vessel',
      'kayak',
      'navigation_buoy',
      'red_lateral_mark',
      'green_lateral_mark',
      'cardinal_mark',
      'special_mark',
      'floating_object',
      'person',
    ];

    const $ = (id) => document.getElementById(id);

    function applyChrome() {
      // Immersive layout is always on; chrome is handled by the floating top bar.
    }

    function clearFocusedCamera() {
      $('app').classList.remove('focused');
      for (const card of document.querySelectorAll('.camera.focused')) {
        card.classList.remove('focused');
      }
    }

    function toggleFocus(cameraId) {
      if (!config) return;
      const current = document.querySelector('.camera.focused');
      if (current && current.dataset.cameraId === cameraId) {
        clearFocusedCamera();
        return;
      }
      for (const card of document.querySelectorAll('.camera')) {
        card.classList.toggle('focused', card.dataset.cameraId === cameraId);
      }
      $('app').classList.add('focused');
    }

    function columnsToCss(columns, minWidth) {
      if (columns === '1' || columns === '2' || columns === '3') {
        return `repeat(${columns}, minmax(0, 1fr))`;
      }
      return `repeat(auto-fit, minmax(${minWidth || 420}px, 1fr))`;
    }

    function eventLine(event) {
      const source = event.demo ? 'DEMO' : 'YOLO';
      return `${eventBadge(event.class_name)}  ${event.confidence.toFixed(2)}  ${source}  ${event.class_name}  frame ${event.frame_index}`;
    }

    function eventBadge(className) {
      const badges = {
        boat: 'BOAT',
        ship: 'SHIP',
        sailboat: 'SAIL',
        small_vessel: 'SMV',
        kayak: 'KAYK',
        navigation_buoy: 'BUOY',
        red_lateral_mark: 'RED',
        green_lateral_mark: 'GRN',
        cardinal_mark: 'CARD',
        special_mark: 'SPEC',
        floating_object: 'OBJ',
        person: 'PERS',
      };
      return badges[className] || className.slice(0, 4).toUpperCase();
    }

    function zoneText(zones) {
      return (zones || []).map((zone) =>
        [zone.x1, zone.y1, zone.x2, zone.y2].map((value) => Number(value).toFixed(2)).join(', ')
      ).join('\n');
    }

    function parseZones(text) {
      return text.split('\n').map((line) => line.trim()).filter(Boolean).map((line) => {
        const values = line.split(',').map((value) => Number(value.trim()));
        if (values.length !== 4 || values.some((value) => Number.isNaN(value))) {
          throw new Error('Ignore zones must use: x1, y1, x2, y2');
        }
        return {
          x1: Math.max(0, Math.min(1, values[0])),
          y1: Math.max(0, Math.min(1, values[1])),
          x2: Math.max(0, Math.min(1, values[2])),
          y2: Math.max(0, Math.min(1, values[3])),
        };
      });
    }

    function applyLayout() {
      if (!config) return;
      $('camera-grid').style.gridTemplateColumns = columnsToCss(config.dashboard.columns, config.dashboard.card_min_width);
      $('app').classList.toggle('hide-labels', config.dashboard.show_labels === false);
      $('app').classList.toggle('hide-status', config.dashboard.show_status === false);
      for (const pre of document.querySelectorAll('pre.events')) {
        pre.style.display = config.dashboard.show_events ? 'block' : 'none';
      }
      applyChrome();
    }

    function renderCameraCards(status) {
      const grid = $('camera-grid');
      const existing = new Set([...grid.querySelectorAll('.camera')].map((node) => node.dataset.cameraId));
      for (const camera of status.cameras) {
        if (!existing.has(camera.camera_id)) {
          const section = document.createElement('section');
          section.className = 'camera';
          section.dataset.cameraId = camera.camera_id;
          section.innerHTML = `
            <div class="camera-header">
              <h2></h2>
              <span class="status"></span>
              <button class="cam-ai" title="Turn AI on/off for this feed"><i class="fa-solid fa-brain"></i> <span class="cam-ai-label">AI</span></button>
            </div>
            <img alt="">
            <div class="overlay"></div>
            <pre class="events"></pre>
          `;
          section.addEventListener('click', (e) => {
            if (e.target.closest('.cam-ai')) return;  // let the AI toggle handle itself
            toggleFocus(camera.camera_id);
          });
          const aiBtn = section.querySelector('.cam-ai');
          aiBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const enable = aiBtn.classList.contains('off');  // off -> turn on
            aiBtn.classList.toggle('off', !enable);          // optimistic
            try {
              const res = await fetch('/detection.json', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({camera_id: camera.camera_id, enabled: enable}),
              });
              const r = await res.json();
              aiBtn.classList.toggle('off', !r.detect);
              // keep the in-memory config in sync so a later Settings save won't revert it
              const cc = ((config && config.cameras) || []).find((c) => c.camera_id === camera.camera_id);
              if (cc) cc.detect = r.detect;
              showToast(`AI ${r.detect ? 'on' : 'off'} · ${camera.name || camera.camera_id}`);
            } catch (err) { aiBtn.classList.toggle('off', enable); }
          });
          grid.appendChild(section);
        }
      }
      for (const section of [...grid.querySelectorAll('.camera')]) {
        if (!status.cameras.some((camera) => camera.camera_id === section.dataset.cameraId)) {
          section.remove();
        }
      }
      applyLayout();
    }

    const classIcons = {
      boat: 'fa-solid fa-ship',
      ship: 'fa-solid fa-ferry',
      sailboat: 'fa-solid fa-sailboat',
      small_vessel: 'fa-solid fa-ship',
      kayak: 'fa-solid fa-person-paddling',
      navigation_buoy: 'fa-solid fa-circle-dot',
      red_lateral_mark: 'fa-solid fa-location-dot',
      green_lateral_mark: 'fa-solid fa-location-dot',
      cardinal_mark: 'fa-solid fa-compass',
      special_mark: 'fa-solid fa-star',
      floating_object: 'fa-solid fa-box',
      person: 'fa-solid fa-person-swimming',
    };
    function classIcon(className) {
      return classIcons[className] || 'fa-solid fa-location-crosshairs';
    }

    // Alert categories: person & marks always alert on a new one; vessels only
    // when fewer than 3 are present. Floating objects do not alert.
    function alertCategory(name) {
      if (name === 'person') return 'person';
      if (['navigation_buoy','red_lateral_mark','green_lateral_mark','cardinal_mark','special_mark'].includes(name)) return 'mark';
      if (['boat','ship','sailboat','small_vessel','kayak'].includes(name)) return 'vessel';
      return null;
    }
    function checkAlerts(status) {
      if (!alertSoundOn) return;
      let trigger = false;
      for (const cam of status.cameras) {
        const counts = {person: 0, mark: 0, vessel: 0};
        for (const d of (cam.detections || [])) {
          const cat = alertCategory(d.class_name);
          if (cat) counts[cat]++;
        }
        const prev = prevCatCounts[cam.camera_id] || {person: 0, mark: 0, vessel: 0};
        if (counts.person > prev.person) trigger = true;
        if (counts.mark > prev.mark) trigger = true;
        if (counts.vessel > prev.vessel && counts.vessel < 3) trigger = true;
        prevCatCounts[cam.camera_id] = counts;
      }
      const now = Date.now();
      if (trigger && now - lastAlertTime > 1500) {
        lastAlertTime = now;
        try { alertAudio.currentTime = 0; alertAudio.play().catch(() => {}); } catch (e) {}
      }
    }

    function renderMarkers(card, camera) {
      const overlay = card.querySelector('.overlay');
      if (!overlay) return;
      const img = card.querySelector('img');
      const dets = camera.detections || [];
      if (!camera.image_size || camera.status === 'disabled' || !img.clientWidth) {
        overlay.innerHTML = '';
        return;
      }
      // Map normalized coords onto the object-fit: contain content box.
      const iw = camera.image_size.width, ih = camera.image_size.height;
      const ew = img.clientWidth, eh = img.clientHeight;
      const scale = Math.min(ew / iw, eh / ih);
      const dw = iw * scale, dh = ih * scale;
      const offX = (ew - dw) / 2, offY = (eh - dh) / 2;
      overlay.innerHTML = dets.map((d) => {
        const px = offX + d.cx * dw;
        const py = offY + d.y1 * dh;
        const conf = d.confidence ? ` ${d.confidence.toFixed(2)}` : '';
        return `<div class="marker ${d.severity}" style="left:${px.toFixed(1)}px; top:${py.toFixed(1)}px">
          <span class="badge"><i class="${classIcon(d.class_name)}"></i></span>
          <span class="tag">${d.label}${conf}</span>
          <span class="leader"></span>
        </div>`;
      }).join('');
    }

    function updateCameraCards(status) {
      if (typeof status.detection_enabled === 'boolean') reflectAi(status.detection_enabled);
      checkAlerts(status);
      renderCameraCards(status);
      for (const camera of status.cameras) {
        const card = document.querySelector(`.camera[data-camera-id="${CSS.escape(camera.camera_id)}"]`);
        if (!card) continue;
        card.querySelector('h2').textContent = camera.name;
        const statusNode = card.querySelector('.status');
        statusNode.textContent = `${camera.status} · frame ${camera.frame_index}`;
        statusNode.className = `status ${camera.status === 'disabled' ? 'disabled' : ''} ${camera.status.includes('error') ? 'error' : ''}`;
        const aiBtn = card.querySelector('.cam-ai');
        if (aiBtn) {
          aiBtn.classList.toggle('off', camera.detect === false);
          aiBtn.querySelector('.cam-ai-label').textContent = camera.detect === false ? 'AI off' : 'AI';
        }
        const img = card.querySelector('img');
        if (camera.status !== 'disabled') {
          liveCameras.add(camera.camera_id);
        } else {
          liveCameras.delete(camera.camera_id);
          img.removeAttribute('src');
        }
        renderMarkers(card, camera);
        card.querySelector('pre').textContent = camera.events.map(eventLine).join('\n');
      }
    }

    // Show video by rapidly refreshing a still image. This works in every
    // renderer (browser AND the native window's WebView2, which does not render
    // multipart MJPEG streams). Double-buffered to avoid flicker.
    const liveCameras = new Set();
    function pollImages() {
      for (const id of liveCameras) {
        const card = document.querySelector(`.camera[data-camera-id="${CSS.escape(id)}"]`);
        if (!card) continue;
        const img = card.querySelector('img');
        const next = new Image();
        next.onload = () => { img.src = next.src; };
        next.src = `/snapshot.jpg?camera_id=${encodeURIComponent(id)}&_=${Date.now()}`;
      }
    }

    function cameraEditor(camera, index) {
      const wrapper = document.createElement('div');
      wrapper.className = 'camera-config';
      wrapper.dataset.index = index;
      wrapper.innerHTML = `
        <div class="camera-config-head">
          <label class="check"><input class="camera-enabled" type="checkbox"> Enabled</label>
          <button class="danger camera-remove" title="Remove camera">Remove</button>
        </div>
        <div class="row">
          <label>Camera ID<input class="camera-id"></label>
          <label>Name<input class="camera-name"></label>
        </div>
        <label>RTSP, HTTP, or local video source<input class="camera-source"></label>
        <label>Allowed classes<textarea class="camera-classes" placeholder="boat, person"></textarea></label>
        <input class="camera-polygons" type="hidden">
        <input class="camera-ignore-zones" type="hidden">
        <div class="actions">
          <button class="mark-zone">Mark own vessel area</button>
          <span class="poly-info hint"></span>
          <button class="capture-label">Save training label</button>
        </div>
      `;
      wrapper.querySelector('.camera-enabled').checked = camera.enabled !== false;
      wrapper.querySelector('.camera-id').value = camera.camera_id || '';
      wrapper.querySelector('.camera-name').value = camera.name || '';
      wrapper.querySelector('.camera-source').value = camera.source || '';
      wrapper.querySelector('.camera-classes').value = (camera.allowed_classes || []).join(', ');
      wrapper.querySelector('.camera-ignore-zones').value = zoneText(camera.ignore_zones || []);
      wrapper.querySelector('.camera-polygons').value = JSON.stringify(camera.ignore_polygons || []);
      const polyInfo = wrapper.querySelector('.poly-info');
      polyInfo.textContent = describePolys(camera.ignore_polygons || []);
      wrapper.querySelector('.mark-zone').addEventListener('click', () => {
        openZoneModal(
          wrapper.querySelector('.camera-id').value.trim(),
          wrapper.querySelector('.camera-polygons'),
          polyInfo
        );
      });
      wrapper.querySelector('.capture-label').addEventListener('click', () => {
        openLabelModal(wrapper.querySelector('.camera-id').value.trim());
      });
      wrapper.querySelector('.camera-remove').addEventListener('click', () => {
        config.cameras.splice(index, 1);
        renderConfigForm();
      });
      return wrapper;
    }

    function renderConfigForm() {
      $('model').value = config.app.model;
      $('device').value = config.app.device || '';
      $('image-size').value = config.app.image_size;
      $('confidence').value = config.app.confidence_threshold;
      $('jpeg-quality').value = config.app.jpeg_quality;
      $('demo-findings').checked = config.app.demo_findings;
      $('output-jsonl').value = config.app.output_jsonl;
      $('auto-capture').checked = !!config.app.auto_capture;
      $('auto-capture-interval').value = config.app.auto_capture_interval != null ? config.app.auto_capture_interval : 10;
      $('auto-capture-max').value = config.app.auto_capture_max != null ? config.app.auto_capture_max : 500;
      $('columns').value = config.dashboard.columns;
      $('card-min-width').value = config.dashboard.card_min_width;
      $('show-labels').checked = config.dashboard.show_labels !== false;
      $('show-status').checked = config.dashboard.show_status !== false;
      $('show-vessel-area').checked = config.dashboard.show_vessel_area !== false;
      $('alert-sound').checked = alertSoundOn;
      $('show-events').checked = config.dashboard.show_events;
      $('start-wall-mode').checked = config.dashboard.view_mode === 'wall';

      const holder = $('camera-configs');
      holder.innerHTML = '';
      config.cameras.forEach((camera, index) => holder.appendChild(cameraEditor(camera, index)));
      applyLayout();
    }

    function collectConfig() {
      config.app.model = $('model').value.trim() || 'yolo26n.pt';
      config.app.device = $('device').value.trim();
      config.app.image_size = Number($('image-size').value || 640);
      config.app.confidence_threshold = Number($('confidence').value || 0.35);
      config.app.jpeg_quality = Number($('jpeg-quality').value || 80);
      config.app.demo_findings = $('demo-findings').checked;
      config.app.output_jsonl = $('output-jsonl').value.trim() || 'outputs/events/live_detections.jsonl';
      config.app.auto_capture = $('auto-capture').checked;
      config.app.auto_capture_interval = Number($('auto-capture-interval').value || 10);
      config.app.auto_capture_max = Number($('auto-capture-max').value || 500);
      config.dashboard.columns = $('columns').value;
      config.dashboard.card_min_width = Number($('card-min-width').value || 420);
      config.dashboard.show_labels = $('show-labels').checked;
      config.dashboard.show_status = $('show-status').checked;
      config.dashboard.show_vessel_area = $('show-vessel-area').checked;
      config.dashboard.show_events = $('show-events').checked;
      config.dashboard.view_mode = $('start-wall-mode').checked ? 'wall' : 'workspace';
      config.cameras = [...document.querySelectorAll('.camera-config')].map((node) => {
        let polys = [];
        try { polys = JSON.parse(node.querySelector('.camera-polygons').value || '[]'); } catch (e) {}
        const cid = node.querySelector('.camera-id').value.trim();
        const prev = (config.cameras || []).find((c) => c.camera_id === cid);
        return {
          enabled: node.querySelector('.camera-enabled').checked,
          // preserve the per-feed AI toggle (set from the camera card, not this form)
          detect: prev ? prev.detect !== false : true,
          camera_id: cid,
          name: node.querySelector('.camera-name').value.trim(),
          source: node.querySelector('.camera-source').value.trim(),
          allowed_classes: node.querySelector('.camera-classes').value.split(',').map((item) => item.trim()).filter(Boolean),
          ignore_zones: parseZones(node.querySelector('.camera-ignore-zones').value),
          ignore_polygons: polys,
        };
      });
    }

    async function loadConfig() {
      const response = await fetch('/config.json');
      config = await response.json();
      renderConfigForm();
    }

    async function saveConfig() {
      try {
        collectConfig();
      } catch (error) {
        $('message').textContent = error.message;
        return;
      }
      $('message').textContent = 'Saving...';
      const response = await fetch('/config.json', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(config),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        $('message').textContent = result.error || 'Save failed';
        return;
      }
      $('message').textContent = 'Saved. Camera workers restarted.';
      await loadConfig();
    }

    async function refreshStatus() {
      const response = await fetch('/status.json');
      updateCameraCards(await response.json());
    }

    // ---- Own-vessel area: polygon drawing ----
    let activePolyInput = null, activePolyInfo = null, polyPoints = [];

    function describePolys(polys) {
      if (!polys || !polys.length || !(polys[0] || []).length) return 'No own-vessel area set';
      const n = (polys[0] || []).length;
      return `Own-vessel polygon: ${n} point${n === 1 ? '' : 's'}`;
    }

    function pointInImage(event) {
      const rect = $('zone-image').getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
        y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
      };
    }

    function renderPoly() {
      const svg = $('zone-svg');
      const pts = polyPoints.map((p) => `${(p.x * 100).toFixed(2)},${(p.y * 100).toFixed(2)}`).join(' ');
      let inner = polyPoints.length >= 2 ? `<polygon points="${pts}"></polygon>` : '';
      for (const p of polyPoints) inner += `<circle cx="${(p.x * 100).toFixed(2)}" cy="${(p.y * 100).toFixed(2)}" r="1.2"></circle>`;
      svg.innerHTML = inner;
      $('zone-output').textContent = `${polyPoints.length} point(s) — click to add, need at least 3`;
    }

    function openZoneModal(cameraId, polyInput, polyInfo) {
      if (!cameraId) {
        $('message').textContent = 'Save the camera ID before marking the own-vessel area.';
        return;
      }
      activePolyInput = polyInput;
      activePolyInfo = polyInfo;
      let existing = [];
      try { existing = JSON.parse(polyInput.value || '[]'); } catch (e) {}
      polyPoints = ((existing[0] || [])).map((p) => ({x: p.x, y: p.y}));
      $('zone-image').src = `/snapshot.jpg?camera_id=${encodeURIComponent(cameraId)}&marker=${Date.now()}`;
      renderPoly();
      $('zone-modal').classList.add('open');
    }

    $('zone-canvas').addEventListener('click', (event) => {
      if (event.target.tagName === 'BUTTON') return;
      polyPoints.push(pointInImage(event));
      renderPoly();
    });
    $('zone-undo').addEventListener('click', () => { polyPoints.pop(); renderPoly(); });
    $('zone-apply').addEventListener('click', () => {
      if (!activePolyInput) return;
      activePolyInput.value = polyPoints.length >= 3 ? JSON.stringify([polyPoints]) : '[]';
      if (activePolyInfo) activePolyInfo.textContent = describePolys(JSON.parse(activePolyInput.value));
      $('zone-modal').classList.remove('open');
    });
    $('zone-clear').addEventListener('click', () => { polyPoints = []; renderPoly(); });
    $('zone-close').addEventListener('click', () => { $('zone-modal').classList.remove('open'); });

    function populateLabelClasses() {
      const select = $('label-class');
      select.innerHTML = '';
      for (const className of maritimeClasses) {
        const option = document.createElement('option');
        option.value = className;
        option.textContent = className.replaceAll('_', ' ');
        select.appendChild(option);
      }
      select.value = 'navigation_buoy';
    }

    function showLabelBox(box) {
      const node = $('label-box');
      const canvas = $('label-canvas');
      const width = canvas.clientWidth;
      const height = $('label-image').clientHeight;
      node.style.left = `${box.x1 * width}px`;
      node.style.top = `${box.y1 * height}px`;
      node.style.width = `${(box.x2 - box.x1) * width}px`;
      node.style.height = `${(box.y2 - box.y1) * height}px`;
      node.style.display = 'block';
      $('label-output').textContent = zoneText([box]);
    }

    function pointInLabelImage(event) {
      const image = $('label-image');
      const rect = image.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
        y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
      };
    }

    function openLabelModal(cameraId) {
      if (!cameraId) {
        $('message').textContent = 'Save the camera ID before labeling.';
        return;
      }
      activeLabelCameraId = cameraId;
      activeLabelBox = null;
      $('label-box').style.display = 'none';
      $('label-output').textContent = '';
      populateLabelClasses();
      $('label-image').src = `/raw_snapshot.jpg?camera_id=${encodeURIComponent(cameraId)}&marker=${Date.now()}`;
      $('label-modal').classList.add('open');
    }

    $('label-canvas').addEventListener('mousedown', (event) => {
      labelDragStart = pointInLabelImage(event);
      activeLabelBox = {x1: labelDragStart.x, y1: labelDragStart.y, x2: labelDragStart.x, y2: labelDragStart.y};
      showLabelBox(activeLabelBox);
    });
    window.addEventListener('mousemove', (event) => {
      if (!labelDragStart) return;
      const point = pointInLabelImage(event);
      activeLabelBox = {
        x1: Math.min(labelDragStart.x, point.x),
        y1: Math.min(labelDragStart.y, point.y),
        x2: Math.max(labelDragStart.x, point.x),
        y2: Math.max(labelDragStart.y, point.y),
      };
      showLabelBox(activeLabelBox);
    });
    window.addEventListener('mouseup', () => {
      labelDragStart = null;
    });
    $('label-save').addEventListener('click', async () => {
      if (!activeLabelCameraId || !activeLabelBox) {
        $('label-output').textContent = 'Draw a box first.';
        return;
      }
      const response = await fetch('/annotation.json', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          camera_id: activeLabelCameraId,
          class_name: $('label-class').value,
          bbox: activeLabelBox,
        }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        $('label-output').textContent = result.error || 'Save failed';
        return;
      }
      $('label-output').textContent = `Saved ${result.class_name}: ${result.image}`;
    });
    $('label-close').addEventListener('click', () => {
      $('label-modal').classList.remove('open');
    });
    window.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') clearFocusedCamera();
    });

    $('save').addEventListener('click', saveConfig);
    $('add-camera').addEventListener('click', () => {
      config.cameras.push({
        enabled: true,
        camera_id: `camera_${config.cameras.length + 1}`,
        name: `Camera ${config.cameras.length + 1}`,
        source: 'rtsp://username:password@192.168.x.x:554/stream1',
        allowed_classes: ['boat', 'person'],
        ignore_zones: [],
      });
      renderConfigForm();
    });
    function setSettingsOpen(open) {
      $('settings').classList.toggle('open', open);
      $('settings-scrim').classList.toggle('open', open);
    }
    $('toggle-settings').addEventListener('click', () => {
      setSettingsOpen(!$('settings').classList.contains('open'));
    });
    $('settings-close').addEventListener('click', () => setSettingsOpen(false));
    $('settings-scrim').addEventListener('click', () => setSettingsOpen(false));

    // ---- Auto-hide top bar: reveal only when the cursor is in the top 20% ----
    (function () {
      const bar = document.querySelector('.topbar');
      if (!bar) return;
      const REVEAL_BAND = 0.20;   // top fraction of the window that reveals the bar
      const HIDE_DELAY = 1500;    // ms without qualifying input before it tucks away
      let hideTimer = null;
      const pinned = () =>
        $('settings').classList.contains('open') ||
        $('display-pop').classList.contains('open') ||
        bar.matches(':hover');
      function show() {
        bar.classList.remove('hidden');
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
      }
      function scheduleHide() {
        if (hideTimer) clearTimeout(hideTimer);
        hideTimer = setTimeout(() => { if (!pinned()) bar.classList.add('hidden'); }, HIDE_DELAY);
      }
      window.addEventListener('mousemove', (e) => {
        if (e.clientY <= window.innerHeight * REVEAL_BAND) show();
        else if (!pinned()) scheduleHide();
      });
      bar.addEventListener('mouseenter', show);
      bar.addEventListener('mouseleave', scheduleHide);
      window.addEventListener('touchstart', (e) => {
        const y = (e.touches && e.touches[0]) ? e.touches[0].clientY : 1e9;
        if (y <= window.innerHeight * REVEAL_BAND) { show(); scheduleHide(); }
      }, { passive: true });
      // Reveal briefly on load so the controls are discoverable, then tuck away.
      show();
      scheduleHide();
    })();
    for (const id of ['columns', 'card-min-width', 'show-labels', 'show-status', 'show-events', 'start-wall-mode']) {
      $(id).addEventListener('change', () => {
        collectConfig();
        applyLayout();
      });
    }
    $('alert-sound').addEventListener('change', (e) => {
      alertSoundOn = e.target.checked;
      localStorage.setItem('bv-alert-sound', alertSoundOn ? 'on' : 'off');
      if (alertSoundOn) { try { alertAudio.play().then(() => { alertAudio.pause(); alertAudio.currentTime = 0; }).catch(() => {}); } catch (e) {} }
    });

    // ---- OpenBridge Brilliance + Day/Dusk/Night theme ----
    function applyTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      for (const b of document.querySelectorAll('.theme-seg button')) {
        b.classList.toggle('active', b.dataset.themeVal === theme);
      }
      try { localStorage.setItem('bv-theme', theme); } catch (e) {}
    }
    function applyBrilliance(pct) {
      pct = Math.max(25, Math.min(100, Number(pct) || 100));
      document.documentElement.style.setProperty('--brilliance', (pct / 100).toFixed(2));
      const slider = $('brilliance');
      slider.value = pct;
      slider.style.setProperty('--fill', ((pct - 25) / 75 * 100).toFixed(0) + '%');
      $('brilliance-val').textContent = pct;
      try { localStorage.setItem('bv-brilliance', pct); } catch (e) {}
    }
    let toastTimer = null;
    function showToast(message) {
      const t = $('toast');
      t.textContent = message;
      t.classList.add('show');
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
    }
    let aiEnabled = true;
    function reflectAi(enabled) {
      aiEnabled = enabled;
      $('ai-toggle').classList.toggle('ai-off', !enabled);
      $('ai-label').textContent = enabled ? 'AI on' : 'AI off';
    }
    $('ai-toggle').addEventListener('click', async () => {
      const next = !aiEnabled;
      reflectAi(next); // optimistic
      try {
        const res = await fetch('/detection.json', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({enabled: next}),
        });
        const r = await res.json();
        reflectAi(!!r.detection_enabled);
        showToast(r.detection_enabled ? 'AI detection on' : 'AI detection off');
      } catch (e) { showToast('Could not toggle AI'); }
    });
    $('capture-btn').addEventListener('click', async () => {
      const focused = document.querySelector('.camera.focused');
      const cameraId = focused ? focused.dataset.cameraId : '';
      const btn = $('capture-btn');
      btn.classList.add('flash');
      setTimeout(() => btn.classList.remove('flash'), 300);
      try {
        const res = await fetch('/capture.json', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(cameraId ? {camera_id: cameraId} : {}),
        });
        const r = await res.json();
        showToast(r.count
          ? `📸 Saved ${r.count} frame${r.count === 1 ? '' : 's'} for labeling`
          : 'No frame available to capture yet');
      } catch (e) {
        showToast('Capture failed');
      }
    });
    $('display-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      $('display-pop').classList.toggle('open');
    });
    document.addEventListener('click', (e) => {
      if (!$('display-pop').contains(e.target) && e.target !== $('display-btn')) {
        $('display-pop').classList.remove('open');
      }
    });
    for (const b of document.querySelectorAll('.theme-seg button')) {
      b.addEventListener('click', () => applyTheme(b.dataset.themeVal));
    }
    $('brilliance').addEventListener('input', (e) => applyBrilliance(e.target.value));
    applyTheme(localStorage.getItem('bv-theme') || 'day');
    applyBrilliance(localStorage.getItem('bv-brilliance') || 100);

    loadConfig().then(() => {
      refreshStatus();
      setInterval(refreshStatus, 500);
      setInterval(pollImages, 50);
    });
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local browser dashboard for live RTSP YOLO detection.")
    parser.add_argument("--config", default="configs/windows_cameras.local.yaml")
    return parser.parse_args()


def main() -> int:
    global STATE
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    STATE = DashboardState(config_path, config)

    server = ThreadingHTTPServer((config.host, config.port), DashboardHandler)
    url = f"http://{config.host}:{config.port}"
    print(f"Boat Vision dashboard: {url}")
    print(f"Config file: {config_path}")
    print(f"Writing events to: {config.output_jsonl}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
