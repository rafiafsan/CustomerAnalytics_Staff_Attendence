"""OmniRetail AI Suite — Standalone Master Desktop GUI Launcher
============================================================
Pure graphical desktop application launcher with zero terminal window (console=False).
Integrates:
- FootFall_AgeGender: Customer In/Out, Store Occupancy, Gate Calibration, Demographics
- VisionGuard AI: 8-Stage Face Landmark Attendance, Quality Filter, SFace Verification, Staff Roster
"""

import os
import sys
import types
import importlib.util
from pathlib import Path

# Force TCP RTSP and CPU Inference defaults
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

# Resolve Workspace and Subproject Roots (auto-detects root or subfolder execution)
current_dir = Path(__file__).resolve().parent
if (current_dir / "FootFall_AgeGender").exists() and (current_dir / "Attendence System").exists():
    PROJECT_ROOT = current_dir
elif (current_dir.parent / "FootFall_AgeGender").exists() and (current_dir.parent / "Attendence System").exists():
    PROJECT_ROOT = current_dir.parent
else:
    PROJECT_ROOT = current_dir

FOOTFALL_ROOT = PROJECT_ROOT / "FootFall_AgeGender"
ATTENDANCE_ROOT = PROJECT_ROOT / "Attendence System"

# Ensure working directory is set to FootFall_AgeGender for relative model and config resolution
os.chdir(FOOTFALL_ROOT)

# Add directories to sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FOOTFALL_ROOT))
sys.path.insert(0, str(ATTENDANCE_ROOT))
sys.path.insert(0, str(FOOTFALL_ROOT / "mivolo_system"))

# -----------------------------------------------------------------------------
# Module Namespace Bridge (Resolves config, database, and gui across both apps)
# -----------------------------------------------------------------------------
# 1. Bridge 'config': Provide VisionGuard config attributes while exposing FootFall config directory
spec_cfg = importlib.util.spec_from_file_location("config", ATTENDANCE_ROOT / "config.py")
mod_cfg = importlib.util.module_from_spec(spec_cfg)
spec_cfg.loader.exec_module(mod_cfg)
mod_cfg.__path__ = [str(FOOTFALL_ROOT / "config")]
sys.modules["config"] = mod_cfg

# 2. Bridge 'database': Provide VisionGuard DatabaseManager while exposing FootFall database package
spec_db = importlib.util.spec_from_file_location("database", ATTENDANCE_ROOT / "database.py")
mod_db = importlib.util.module_from_spec(spec_db)
spec_db.loader.exec_module(mod_db)
mod_db.__path__ = [str(FOOTFALL_ROOT / "database")]
sys.modules["database"] = mod_db

# 3. Bridge 'gui': Combined namespace mapping for retail_gui and attendance_gui
mod_gui = types.ModuleType("gui")
mod_gui.__path__ = [str(FOOTFALL_ROOT / "gui"), str(ATTENDANCE_ROOT / "gui")]
sys.modules["gui"] = mod_gui

from unified_gui import launch_unified_gui


def main():
    launch_unified_gui()


if __name__ == "__main__":
    main()
