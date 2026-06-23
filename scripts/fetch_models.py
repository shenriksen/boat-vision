"""CI helper: make sure the base YOLO weights exist in the repo root.

`*.pt` is gitignored, so a fresh checkout (e.g. a GitHub Actions runner) has no
weights — but the PyInstaller build bundles `yolo26n.pt` / `yolo26s.pt` via
`--add-data`. Ultralytics downloads any missing official weights on first load;
this script triggers that and copies them into the repo root so the build finds them.
"""
from __future__ import annotations

import os
import shutil
import sys

from ultralytics import YOLO

MODELS = ("yolo26n.pt", "yolo26s.pt")


def main() -> int:
    for name in MODELS:
        model = YOLO(name)  # downloads if missing
        src = getattr(model, "ckpt_path", None) or name
        if src and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(name):
            shutil.copy(src, name)
        if not os.path.exists(name):
            print(f"ERROR: could not obtain {name}", file=sys.stderr)
            return 1
        print(f"ready: {name} ({os.path.getsize(name) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
