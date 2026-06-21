# Data Storage Plan

Use the local project folder for active capture/training/runtime work, and sync
curated data to OneDrive for backup and sharing. The final runtime target is the
Windows boat PC.

## Recommended OneDrive Folder

On this Mac, OneDrive is available at:

```text
/Users/sanderhenriksen/Library/CloudStorage/OneDrive-NordicUsvAS
```

On Windows, OneDrive is usually one of:

```text
C:\Users\<you>\OneDrive - Nordic Usv AS
C:\Users\<you>\OneDrive
```

Recommended project folder:

```text
/Users/sanderhenriksen/Library/CloudStorage/OneDrive-NordicUsvAS/BoatVisionDataset
```

## What To Store

Store these in OneDrive:

```text
BoatVisionDataset/
  raw_clips/
    bow/
    stern/
    port/
    starboard/
  datasets/
    maritime/
      images/
      labels/
      raw_frames/
  models/
    maritime/
      best.pt
  events/
```

Keep these local on the Windows boat PC for runtime:

```text
configs/windows_cameras.local.yaml
models/maritime/best.pt
outputs/events/
```

## Why Not Run Directly From OneDrive

For live onboard use, do not run video inference directly from a synced OneDrive
folder. Sync clients can pause, lock files, create partial files, or use
cloud-only placeholders.

Use OneDrive for:

- Backup.
- Sharing clips for labeling.
- Moving trained model artifacts between machines.

Use the local disk for:

- RTSP processing.
- Live event output.
- The active model file.

## Export Current Dataset Artifacts

From the project root:

macOS:

```sh
bash scripts/export_dataset_to_onedrive.sh
```

Windows:

```powershell
.\scripts\export_dataset_to_onedrive.ps1
```

Optional custom destination on Windows:

```powershell
.\scripts\export_dataset_to_onedrive.ps1 "C:\Users\<you>\OneDrive - Nordic Usv AS\BoatVisionDataset"
```

Optional custom destination on macOS:

```sh
bash scripts/export_dataset_to_onedrive.sh "/path/to/OneDrive/BoatVisionDataset"
```

## Live Labeling Workflow

The dashboard can save training labels:

1. Open Settings.
2. Choose a camera.
3. Click `Save training label`.
4. Select class, for example `navigation_buoy`.
5. Draw a box.
6. Save.

The app writes:

```text
data/datasets/maritime/images/train/*.jpg
data/datasets/maritime/labels/train/*.txt
```

Then export to OneDrive with the script above.
