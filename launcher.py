"""Boat Vision desktop launcher.

Builds into BoatVision.exe (via PyInstaller). Double-clicking it:
  1. starts the dashboard server in the background (no console window),
  2. waits for it to come up,
  3. opens Boat Vision in a clean app window (Edge app-mode), or the default
     browser as a fallback.

The .exe is a thin launcher: it runs the already-installed app in .venv, so it
stays small (it does NOT bundle PyTorch/CUDA).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

URL = "http://127.0.0.1:8765"


def app_root() -> Path:
    # When frozen as BoatVision.exe, use the folder the .exe sits in.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    root = app_root()
    python = root / ".venv" / "Scripts" / "python.exe"
    config = root / "configs" / "windows_cameras.local.yaml"
    example = root / "configs" / "windows_cameras.example.yaml"

    if not python.exists():
        print("Could not find .venv. Run scripts\\setup_windows.ps1 first.")
        return 1

    if not config.exists() and example.exists():
        config.write_bytes(example.read_bytes())

    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    subprocess.Popen(
        [str(python), "-m", "boat_vision.live_dashboard", "--config", str(config)],
        cwd=str(root),
        creationflags=creationflags,
    )

    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "/status.json", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    for edge in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(edge).exists():
            subprocess.Popen([edge, "--app=" + URL, "--window-size=1400,900"])
            return 0
    webbrowser.open(URL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
