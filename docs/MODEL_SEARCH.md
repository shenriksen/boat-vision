# Custom Maritime Model Search

The generic pretrained YOLO model does not reliably detect navigation buoys,
seamarks, or lateral/cardinal marks. It missed the buoy in the supplied rear
view clip even when tested with lower confidence and larger image size.

## Reality check on class granularity

There is an important gap to understand before downloading anything:

- **Public datasets almost always have a single coarse `buoy` class.** They do
  NOT separate IALA types (red/green lateral, cardinal, isolated danger, safe
  water, special). Fine-grained IALA discrimination is also region-specific
  (Norway is IALA **Region A**), and no public dataset covers that well.
- **Therefore the bootstrap goal is detection, not classification.** Use public
  data to get a model that reliably *finds* a buoy/seamark (one class). Splitting
  into the full IALA taxonomy comes later, from your own labeled footage.

Recommended bootstrap taxonomy: train first with a single `seamark` (or
`navigation_buoy`) class plus `boat`/`person`. Expand to the full IALA class
list in `configs/maritime_dataset.yaml` once you have enough of your own labels.

## Candidate Sources (verified June 2026)

### 1. Roboflow Universe — "Maritime YOLO" by earsdataset  ⭐ best bootstrap

- URL: https://universe.roboflow.com/earsdataset-vxvbd/maritime-yolo-iqrc2
- Classes include: `boat, ship, buoy, anchor, lighthouse`
- **License: CC BY 4.0** (commercial use allowed with attribution)
- Already in YOLO format, one-click "Export" → Ultralytics/YOLOv8 download.
- Easiest path: no parsing, drops straight into a `data.yaml` Ultralytics can
  train on. Requires a free Roboflow account + API key to download.

### 2. Roboflow Universe — "buoy" by YOLO project

- URL: https://universe.roboflow.com/yolo-project/buoy
- ~391 buoy images, buoy-focused. Small but clean, good as supplemental data.
- Check the per-dataset license on the page before commercial use.

### 3. Singapore Maritime Dataset Plus (SMD-Plus)

- GitHub: https://github.com/kjunhwa/Singapore-Maritime-Dataset-Plus
- Real on-water video with corrected annotations. 7 classes incl. **Buoy**,
  Vessel/ship, Boat, Kayak, Sail boat, Ferry.
- Strong real-world maritime footage; academic/research provenance — confirm
  license terms before commercial deployment. Annotations need converting to
  YOLO txt format.

### 4. MODD-13

- 9000+ annotated images, explicitly includes **buoys and lighthouses** for USV
  navigation (most public sets ignore these). Ideal in scope.
- Caveat: no easy public download link found; it is paper-gated (Wang et al.,
  2024). Would require contacting the authors.

### 5. KOLOMVERSE

- Large-scale Korean maritime dataset: `ship, buoy, fishnet buoy, lighthouse,
  wind farm`. Useful for pretraining/eval; classes don't match IALA marks.

## Recommended Path

1. **Bootstrap now:** download dataset #1 (Maritime YOLO, CC BY 4.0) in YOLO
   format and train a coarse buoy/seamark + boat + person detector. This gets a
   model that actually puts a box on a buoy — the thing the pretrained model
   failed to do.
2. **Validate** on your own clips (`IMG_5134.MOV`, `IMG_5136.MOV`) — these were
   NOT in training, so they are a fair test.
3. **Collect** frames from your real cameras using the dashboard / passive
   capture, especially the exact buoy/seamark types in your operating area.
4. **Fine-tune**: add your own labeled frames and (optionally) datasets #2/#3 as
   supplemental data, then retrain. Expand to full IALA classes here.
5. Train a local Ultralytics model:

```sh
python -m boat_vision.train --data configs/maritime_dataset.yaml --model yolo26n.pt --epochs 100 --imgsz 960
```

5. Deploy the trained local model on Windows:

```powershell
python -m boat_vision.live_dashboard --config configs\windows_cameras.local.yaml
```

with:

```yaml
app:
  model: models/maritime/best.pt
```

## Current Local Training Frames

Frames have been extracted from the Stern buoy clip to:

```text
data/datasets/maritime/raw_frames/stern_buoy
```

Label the visible buoy/seamark in those frames, then export YOLO labels into:

```text
data/datasets/maritime/images/train
data/datasets/maritime/images/val
data/datasets/maritime/labels/train
data/datasets/maritime/labels/val
```

Train:

```sh
python -m boat_vision.train --data configs/maritime_dataset.yaml --model yolo26n.pt --epochs 100 --imgsz 960
```

Install the trained model for the dashboard:

```sh
mkdir -p models/maritime
cp runs/maritime/custom_yolo/weights/best.pt models/maritime/best.pt
```

## Why Not Use Cloud Inference

The target system should work onboard, locally, without cloud video processing.
Cloud APIs are useful for testing and comparison, but they are not the
deployment architecture.
