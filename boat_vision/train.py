from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import torch
from ultralytics import YOLO


def pick_device(requested: str) -> str:
    """Auto-select the best available device unless one was given."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_batch(requested: str, device: str) -> Union[int, float]:
    """Resolve --batch. 'auto' uses CUDA auto-batch (-1) or a safe size elsewhere."""
    if requested and requested != "auto":
        return float(requested) if "." in requested else int(requested)
    # Ultralytics auto-batch (-1) only works on CUDA; MPS/CPU need an explicit size.
    return -1 if device.isdigit() else 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train / fine-tune the maritime YOLO detector.")
    parser.add_argument("--data", default="configs/maritime_dataset.yaml",
                        help="Dataset YAML. Use configs/singapore_maritime.yaml for the bootstrap dataset.")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Base model or a checkpoint to fine-tune (e.g. models/maritime/best.pt).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", default="auto", help="Integer, float fraction, or 'auto'.")
    parser.add_argument("--device", default="", help="cuda index (0), mps, cpu, or blank to auto-detect.")
    parser.add_argument("--project", default="runs/maritime")
    parser.add_argument("--name", default="custom_yolo")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deploy", dest="deploy", action="store_true", default=True,
                        help="Copy the trained best.pt to models/maritime/best.pt (default on).")
    parser.add_argument("--no-deploy", dest="deploy", action="store_false")
    return parser.parse_args()


def deploy_model(best: Path) -> None:
    target_dir = Path("models/maritime")
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    versioned = target_dir / f"best_{stamp}.pt"
    shutil.copy2(best, versioned)
    shutil.copy2(best, target_dir / "best.pt")
    print(f"\nDeployed model:\n  {target_dir / 'best.pt'}  (active, auto-loaded by the dashboard)\n  {versioned}  (versioned backup)")


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    batch = pick_batch(args.batch, device)

    if device in {"cpu", "mps"}:
        print(f"WARNING: training on '{device}'. This is slow and (on a Mac) should not run while the\n"
              f"         live dashboard is using the GPU. Prefer an NVIDIA GPU (device 0) or the cloud.\n")

    print(f"Training: data={args.data} model={args.model} device={device} batch={batch} "
          f"imgsz={args.imgsz} epochs={args.epochs}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    save_dir = Path(getattr(results, "save_dir", "") or model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        print(f"\nBest weights: {best}")
        if args.deploy:
            deploy_model(best)
    else:
        print("Training finished but best.pt was not found; nothing deployed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
