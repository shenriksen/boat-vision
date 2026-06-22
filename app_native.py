"""Boat Vision - native desktop app (no browser).

Runs the detection dashboard server in-process and shows it in a native
window via pywebview (Edge WebView2). Bundled into BoatVision.exe.

Writable data (configs, captured frames, event logs, trained models) lives in
%LOCALAPPDATA%\\BoatVision so it works whether launched from an installed
location or a portable folder.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import urllib.request
import json
from pathlib import Path

REPO = "shenriksen/boat-vision"
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"


def bundle_dir() -> Path:
    # Files bundled by PyInstaller live under _MEIPASS when frozen.
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def app_version() -> str:
    try:
        return (bundle_dir() / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / "BoatVision"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_data(data: Path) -> Path:
    """Lay out the writable working directory and return the config path."""
    bundle = bundle_dir()
    for sub in ("configs", "models/maritime", "data/datasets/maritime/raw_frames",
                "outputs/events", "outputs/annotated"):
        (data / sub).mkdir(parents=True, exist_ok=True)

    # Bundled default model(s) copied in on first run.
    for weights in ("yolo26s.pt", "yolo26n.pt"):
        dst = data / weights
        if not dst.exists() and (bundle / weights).exists():
            shutil.copy2(bundle / weights, dst)

    example = data / "configs" / "windows_cameras.example.yaml"
    if (bundle / "configs" / "windows_cameras.example.yaml").exists():
        shutil.copy2(bundle / "configs" / "windows_cameras.example.yaml", example)

    config = data / "configs" / "windows_cameras.local.yaml"
    if not config.exists() and example.exists():
        shutil.copy2(example, config)
    return config


def start_server(config_path: Path) -> None:
    from boat_vision import live_dashboard as d
    import torch

    config = d.load_config(config_path)
    # The frozen standalone ships CPU PyTorch, so always use CPU there. The GPU
    # build runs this un-frozen (in a venv) and keeps the configured device.
    if getattr(sys, "frozen", False):
        config.device = "cpu"
    elif not torch.cuda.is_available():
        config.device = "cpu"
    config.host = "127.0.0.1"
    config.port = PORT
    d.STATE = d.DashboardState(config_path, config)
    server = d.ThreadingHTTPServer((config.host, config.port), d.DashboardHandler)
    server.serve_forever()


def check_update() -> str | None:
    """Return a newer version string if the latest GitHub release is newer."""
    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "BoatVision"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            tag = json.load(resp).get("tag_name", "").lstrip("v")
    except Exception:
        return None

    def parts(v):
        out = []
        for p in v.split("."):
            try:
                out.append(int(p))
            except ValueError:
                out.append(0)
        return out

    cur = app_version().lstrip("v")
    if tag and parts(tag) > parts(cur):
        return tag
    return None


def maybe_prompt_update() -> None:
    newer = check_update()
    if not newer:
        return
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        if messagebox.askyesno(
            "Boat Vision update available",
            f"A newer version ({newer}) is available.\n\nOpen the download page now?",
        ):
            import webbrowser
            webbrowser.open(f"https://github.com/{REPO}/releases/latest")
        root.destroy()
    except Exception:
        pass


def icon_path() -> Path:
    try:
        from boat_vision import live_dashboard as d
        return Path(d.__file__).resolve().parent / "static" / "app_icon.ico"
    except Exception:
        return bundle_dir() / "boat_vision" / "static" / "app_icon.ico"


def set_taskbar_icon() -> None:
    """Give the window (and taskbar) the Boat Vision logo instead of the generic
    Python icon. Windows-only; runs in a thread and waits for the window."""
    if os.name != "nt":
        return
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NordicUSV.BoatVision")
    except Exception:
        pass
    user32 = ctypes.windll.user32
    ico = str(icon_path())
    WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
    IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x0010, 0x0040
    hicon = user32.LoadImageW(None, ico, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
    if not hicon:
        return
    title = f"Boat Vision  v{app_version()}"
    for _ in range(120):
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
            return
        time.sleep(0.5)


def main() -> int:
    data = data_dir()
    config_path = prepare_data(data)
    os.chdir(data)  # the server uses paths relative to the working dir

    threading.Thread(target=start_server, args=(config_path,), daemon=True).start()

    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "/status.json", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    threading.Thread(target=maybe_prompt_update, daemon=True).start()
    threading.Thread(target=set_taskbar_icon, daemon=True).start()

    # Native window (Edge WebView2). If that is unavailable for any reason,
    # fall back to the default browser so the user still gets the dashboard.
    try:
        import webview
        webview.create_window(f"Boat Vision  v{app_version()}", URL, width=1480, height=920)
        webview.start()
    except Exception:
        import webbrowser
        webbrowser.open(URL)
        threading.Event().wait()  # keep the server alive
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
