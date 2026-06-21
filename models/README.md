# Local Models

Place trained local model weights here.

Expected custom maritime model path:

```text
models/maritime/best.pt
```

The dashboard configs point at this file. Until it exists, the dashboard falls
back to `yolo26n.pt` so the app still runs.

After training, copy:

```text
runs/maritime/custom_yolo/weights/best.pt
```

to:

```text
models/maritime/best.pt
```
