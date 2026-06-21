# Boat Vision — Deployment & Continuous Improvement Plan

## The core idea: a loop that makes the model better over time

```
        ┌──────────────────────────────────────────────┐
        │                                                │
   ┌────▼─────┐   ┌─────────┐   ┌────────┐   ┌─────────┐ │
   │ 1 CAPTURE│──▶│ 2 LABEL │──▶│ 3 TRAIN│──▶│4 DEPLOY  │─┘
   │ on boat  │   │ offline │   │ on GPU │   │ to boat  │
   └──────────┘   └─────────┘   └────────┘   └─────────┘
```

Every time around the loop, the model gets better at **your** waters, **your**
cameras, and the **specific buoys/seamarks** you operate around. The generic
model can't see buoys today; this loop is how it learns to.

---

## Where we are now

- ✅ Working pipeline + immersive dashboard (Mac, light config `preview_light.yaml`).
- ✅ Bootstrap dataset downloaded: Singapore Maritime (Public Domain), at
  `data/datasets/_downloads/singapore_maritime_v5`, config `configs/singapore_maritime.yaml`.
- ✅ Training entrypoint: `boat_vision/train.py`.
- ✅ Windows scripts: `scripts/setup_windows.ps1`, `scripts/run_windows_dashboard.ps1`.
- ✅ OneDrive export: `scripts/export_dataset_to_onedrive.ps1`.
- ⚠️ The Mac can run the UI but **not** real-time 4×1080p inference or training comfortably.
  Heavy compute belongs on the **boat's Windows PC (NVIDIA GPU)** or the cloud.

---

## Phase 1 — Ship it as a Windows app (deployment)

Goal: the boat PC runs the dashboard against the real RTSP cameras, fast.

1. **Install on the Windows PC** (one time):
   ```powershell
   .\scripts\setup_windows.ps1
   copy configs\windows_cameras.example.yaml configs\windows_cameras.local.yaml
   notepad configs\windows_cameras.local.yaml   # paste the real RTSP URLs
   ```
2. **Use the NVIDIA GPU**: install the CUDA build of PyTorch, set `device: 0` in
   the config. A discrete GPU handles 4×1080p in real time (where the Mac can't).
3. **Run it**:
   ```powershell
   .\scripts\run_windows_dashboard.ps1
   ```
   Open `http://127.0.0.1:8765`.
4. **Autostart (optional)**: a Windows Task Scheduler task so the dashboard
   launches on boot and runs headless on the boat.
5. **Later, one-click `.exe`**: package with PyInstaller so no Python knowledge
   is needed on the boat PC. (Do this once the app is stable — not yet.)

**RTSP credentials** live only in `configs/windows_cameras.local.yaml` (gitignored);
status/event outputs mask them.

---

## Phase 2 — Collect data while you use it (on the boat)

The dashboard's day job is operator awareness; its *second* job is quietly
building a training set so the next model is better. To build:

- **Passive frame capture** — save a frame every N seconds per camera to
  `data/datasets/maritime/raw_frames/<camera>/` while running. (Needs building —
  small addition to the dashboard.)
- **One-click "Capture now"** — operator grabs frames when something interesting
  is in view ("there's a buoy right there").
- **Back it up**: run `scripts/export_dataset_to_onedrive.ps1` to sync collected
  frames/clips off the boat to OneDrive. Do **not** run live inference from OneDrive.

Aim for variety: different light, weather, distance, sea state, and every
buoy/mark type you care about.

---

## Phase 3 — Label the data (offline, at a desk)

Labeling is a separate, calm, batch activity — **not** done live on the boat.

- Use a real labeling tool: **Label Studio**, **CVAT**, or **Roboflow** —
  fast keyboard-driven boxing, hundreds of frames per session.
- Label the classes you actually need (start coarse, e.g. one `seamark` +
  `boat` + `person`; split into IALA types — lateral/cardinal/special — once you
  have enough examples).
- Export in **YOLO format** into:
  ```
  data/datasets/maritime/images/{train,val}
  data/datasets/maritime/labels/{train,val}
  ```
- **Bootstrap**: combine your labels with the Singapore Maritime set so you don't
  start from zero.

Target rough volumes: a few hundred instances per class to start; thousands for
robust detection.

---

## Phase 4 — Train (on the GPU machine, never alongside the live dashboard)

```powershell
python -m boat_vision.train --data configs/maritime_dataset.yaml ^
  --model yolo26n.pt --epochs 100 --imgsz 960 --batch 16 --device 0
```

- Start from the **bootstrap** dataset, then **fine-tune** with your own labeled
  frames added in.
- **Validate on held-out clips** the model never trained on (e.g. your
  `IMG_5134/5136/5139–5142` videos) — that's the honest test.
- Output weights: `runs/.../weights/best.pt`.

Where to train: the boat's Windows GPU (overnight, dashboard off) or a cloud GPU.
**Not** the Mac and the dashboard at the same time — that thrashes the GPU.

---

## Phase 5 — Deploy the new model

1. Copy the new weights to `models/maritime/best.pt` (sync via OneDrive or USB).
2. The dashboard **auto-loads** `models/maritime/best.pt` (falls back to
   `yolo26n.pt` if missing) — no code change needed.
3. Compare before/after on the same validation clips; keep the old model as
   `models/maritime/best_vN.pt` so you can roll back.

---

## Phase 6 — Keep improving (active learning + features)

- **Active learning**: capture the frames where the model is unsure or wrong,
  label those specifically, retrain. This improves fastest per labeling hour.
- **Versioning**: date-stamp datasets and models in OneDrive; record which model
  is deployed on the boat.
- **Then layer on** (each its own milestone): object tracking + confidence
  filtering + event throttling (so it doesn't alert every frame), alerts,
  and finally GPS/heading/camera calibration for approximate bearings &
  georeferenced events.

### Suggested cadence
Collect across a few trips → label a batch → retrain → redeploy. Monthly is a
reasonable rhythm to start; tighten it while the model is still weak.

---

## What's safe to do on the Mac vs the boat PC

| Task | Mac (M3) | Boat PC (NVIDIA) |
|---|---|---|
| UI / design work (`preview_light.yaml`, 1 cam) | ✅ | ✅ |
| 4×1080p real-time inference | ❌ too heavy | ✅ |
| Training | ❌ (or only with dashboard off, slow) | ✅ (overnight, dashboard off) |
| Labeling | ✅ (no GPU needed) | ✅ |
```
