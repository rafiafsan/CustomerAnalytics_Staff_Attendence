"""VisionGuard AI — Standalone Windowed GUI Launcher
===================================================
Pure graphical desktop application launcher with zero terminal window (console=False).
All RTSP streams, face recognition, employee rosters, and embedding verification
occur in the native Windows desktop interface.
"""

import os
import sys
from pathlib import Path

# Force reliable TCP transport for RTSP streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["PYTHONUNBUFFERED"] = "1"

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from gui.attendance_gui import launch_gui


def main():
    launch_gui()


if __name__ == "__main__":
    main()
