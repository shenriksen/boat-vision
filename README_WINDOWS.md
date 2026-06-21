# Boat Vision — Windows PC Setup

Onboard AI camera system: live RTSP camera feeds with YOLO detection, an
immersive operator dashboard, and one-click capture of frames for training.

Everything needed to run is inside this folder. The only thing to download on
the PC is Python (the setup script downloads the rest automatically).

### Downloads needed on the Windows PC
- **Python 3.10+ (required):** https://www.python.org/downloads/
  During install, tick **"Add python.exe to PATH"**.
- **NVIDIA GPU driver (usually already installed on the LOQ):** if the GPU isn't
  detected, get the latest "Game Ready" driver for the RTX 4060:
  https://www.nvidia.com/download/index.aspx
- Everything else (PyTorch/CUDA, Ultralytics, OpenCV, the YOLO model, fonts/icons)
  is bundled or auto-installed by `scripts\setup_windows.ps1` — no manual download.

---

## 1. Install (one time)

Open **PowerShell** in this folder and run:

```powershell
.\scripts\setup_windows.ps1
```

This creates a local environment and installs everything, including the
**GPU (CUDA) build of PyTorch** for your NVIDIA card. No NVIDIA GPU? use:

```powershell
.\scripts\setup_windows.ps1 -Cpu
```

If PowerShell blocks the script, run once:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

The setup prints whether your GPU was detected (`CUDA available: True`).

## 2. Enter your camera URLs

```powershell
copy configs\windows_cameras.example.yaml configs\windows_cameras.local.yaml
notepad configs\windows_cameras.local.yaml
```

Put the real RTSP URL for each camera (same URL VLC uses), e.g.:

```yaml
source: rtsp://username:password@192.168.1.50:554/stream1
```

RTSP usually needs a username/password — include them in the URL as above.
This `*.local.yaml` file stays on the PC and is not shared; the dashboard masks
the password in logs and events.

## 3. Run

Setup puts a **"Boat Vision" icon on your Desktop** — just double-click it
(it starts the program and opens the browser automatically).

Or run it manually:

```powershell
.\scripts\run_windows_dashboard.ps1
```

Then open **http://127.0.0.1:8765** in a browser. In the top bar:
- **AI on/off** — turn the YOLO detection on or off (off = clean video, no GPU use)
- **Capture** — save the current frame(s) for training
- **Display** — Day/Dusk/Night + brightness
- **gear** — settings (cameras, auto-capture, labels)

Click any camera to view it full-screen.

> No cameras handy for a first test? In `windows_cameras.local.yaml` set a
> camera `source: "0"` to use the PC webcam, or point it at a local `.mp4`.

---

## Making the model detect buoys & seamarks (training)

Out of the box it uses the generic `yolo26n.pt` model (bundled), which detects
boats and people but **not** buoys/seamarks. To teach it those:

1. **Collect** frames — enable **Auto-capture** in Settings (saves a frame every
   N seconds) and/or tap **Capture**. Frames land in
   `data\datasets\maritime\raw_frames\<camera>\`.
2. **Back up** to OneDrive: `.\scripts\export_dataset_to_onedrive.ps1`
3. **Label** the frames offline (Roboflow / CVAT / Label Studio) in YOLO format.
4. **Train** (with the dashboard stopped so it gets the full GPU):
   ```powershell
   .\scripts\train.ps1 -Data configs\maritime_dataset.yaml -Model yolo26n.pt
   ```
   It auto-copies the result to `models\maritime\best.pt`.
5. The dashboard **auto-loads** `models\maritime\best.pt` next launch.

See `docs\TRAINING_AND_DEPLOYMENT_PLAN.md` for the full loop.

---

## Folder layout

```
boat_vision\      application code + bundled fonts/icons (works offline)
configs\          camera configs (edit windows_cameras.local.yaml)
scripts\          setup / run / train / onedrive-export PowerShell scripts
models\           yolo26n.pt (bundled) + maritime\best.pt (your trained model)
data\
  datasets\maritime\raw_frames\   <- captured training frames go here
outputs\
  events\         detection logs (JSONL)
docs\             plans and reference
```

## Run on boot (optional)

Use Windows Task Scheduler → "Create Task" → Trigger: *At log on* →
Action: `powershell.exe -File <full path>\scripts\run_windows_dashboard.ps1`.
