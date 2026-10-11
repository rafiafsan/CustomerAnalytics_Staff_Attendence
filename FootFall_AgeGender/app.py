"""MCP Tracking - Low-Resource Retail AI Analytics Engine
======================================================
Re-engineered for Windows Mini PC (Intel CPU, ~2GB RAM, No GPU).
Maintains real-time decoupled RTSP preview, ByteTrack, Footfall,
Event-driven Bag Exit validation, MiVOLO demographics, and PostgreSQL persistence.
"""

import argparse
import io
import logging
import os
import sys
from pathlib import Path

# Force reliable TCP transport for RTSP streams (prevents packet drops / tearing)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Target PC has no GPU
os.environ["YOLO_AUTOINSTALL"] = "0"  # Prevent dynamic pip invocations in frozen binary
os.environ["PYTHONUNBUFFERED"] = "1"   # Force line-buffered stdout in frozen binary

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mivolo_system"))

import yaml

from utils.security import mask_url

from services.unified_pipeline import UnifiedRetailPipeline


class _RotatingFile:
    """Minimal file-like object with size-based rotation (keeps `backups` old files)."""

    def __init__(self, path: str, max_bytes: int = 5 * 1024 * 1024, backups: int = 3):
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self._fh = open(path, "a", encoding="utf-8", buffering=1)

    def _rotate(self):
        self._fh.close()
        for i in range(self.backups - 1, 0, -1):
            src, dst = f"{self.path}.{i}", f"{self.path}.{i + 1}"
            if os.path.exists(src):
                os.replace(src, dst)
        os.replace(self.path, f"{self.path}.1")
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        try:
            if self._fh.tell() > self.max_bytes:
                self._rotate()
        except Exception:
            pass
        return self._fh.write(data)

    def flush(self):
        self._fh.flush()


class _TeeWriter:
    """Wraps a stream and mirrors all writes to a secondary file sink.

    Used to redirect sys.stdout so that every print() call in the pipeline
    is simultaneously written to retail_ai.log regardless of TTY state.
    """

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        if self._primary is not None:
            try:
                self._primary.write(data)
                self._primary.flush()
            except Exception:
                pass
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except Exception:
            pass

    def flush(self):
        if self._primary is not None:
            try:
                self._primary.flush()
            except Exception:
                pass
        try:
            self._secondary.flush()
        except Exception:
            pass

    def fileno(self):
        if self._primary is not None:
            try:
                return self._primary.fileno()
            except Exception:
                pass
        raise io.UnsupportedOperation("fileno")

    def isatty(self):
        return False

    # Delegate any other attribute access to primary
    def __getattr__(self, name):
        return getattr(self._primary, name)


def setup_logging() -> None:
    """Configure logging + stdout tee so ALL output goes to retail_ai.log.

    Critical for frozen PyInstaller exe: sys.stdout is block-buffered (or
    None) when piped.  By opening a log file and tee-ing stdout into it we
    guarantee every print() and logging.* call is persisted immediately.
    """
    import io

    log_file = PROJECT_ROOT / "retail_ai.log"

    # Open the log file early — unbuffered so every line is flushed instantly
    try:
        log_fh = _RotatingFile(str(log_file))
    except Exception as e:
        log_fh = None
        print(f"[App] WARNING: Could not open log file {log_file}: {e}", flush=True)

    # Tee sys.stdout -> log file so all print() calls are captured
    if log_fh is not None:
        sys.stdout = _TeeWriter(sys.stdout, log_fh)
        sys.stderr = _TeeWriter(sys.stderr, log_fh)

    # Standard logging module — file + console
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if log_fh is not None:
        fh = logging.StreamHandler(log_fh)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Console handler via original stderr (before tee wrapping above)
    try:
        ch = logging.StreamHandler(sys.__stderr__)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    except Exception:
        pass

    logging.info("=" * 60)
    logging.info("RetailAI v2 -- Log file: %s", log_file)
    logging.info("=" * 60)


def load_default_config(config_path: str = "config/production.yaml") -> dict:
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def update_rtsp_url_in_config(config_path: str, new_url: str) -> bool:
    """Persist updated RTSP URL into the production YAML configuration file."""
    try:
        cfg_file = Path(config_path)
        if not cfg_file.exists():
            return False
        content = cfg_file.read_text(encoding="utf-8")
        import re
        if re.search(r'^\s*rtsp_url:\s*.*$', content, flags=re.MULTILINE):
            new_content = re.sub(
                r'^(\s*rtsp_url:\s*).*$',
                rf'\g<1>"{new_url}"',
                content,
                flags=re.MULTILINE,
            )
            cfg_file.write_text(new_content, encoding="utf-8")
            print(f"[App] Saved updated RTSP URL to {config_path}")
            return True
        else:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "camera" not in data:
                data["camera"] = {}
            data["camera"]["rtsp_url"] = new_url
            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"[App] Saved updated RTSP URL to {config_path}")
            return True
    except Exception as exc:
        print(f"[App] Warning: Failed to update config file: {exc}")
        return False


def parse_args():
    cfg = load_default_config()
    default_source = (
        cfg.get("camera", {}).get("rtsp_url")
        or cfg.get("camera", {}).get("fallback_video")
        or "test_videos/Outlet_Sample_1_Banasree.avi"
    )
    default_model = cfg.get("inference", {}).get("model", "models/yolov8m.onnx")

    parser = argparse.ArgumentParser(description="Retail Edge AI Analytics Engine (Low-Resource Re-engineered)")
    parser.add_argument(
        "--source",
        default=default_source,
        help="Video source: RTSP stream URL, local video file path, or USB camera index (e.g. 0).",
    )
    parser.add_argument(
        "--config",
        default="config/production.yaml",
        help="Path to production YAML configuration file.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive prompt for NVR RTSP stream and gate calibration.",
    )
    parser.add_argument(
        "--weights",
        default=default_model,
        help="Path to primary YOLO model (ONNX or PyTorch weights).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch the interactive multi-detector gate calibration tool before running.",
    )
    parser.add_argument(
        "--no-demographics",
        action="store_true",
        help="Disable the asynchronous MiVOLO age/gender demographic profiling worker.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without opening an OpenCV preview window.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Force opening an OpenCV preview window.",
    )
    parser.add_argument(
        "--service",
        action="store_true",
        help="Run unattended in background as a Windows service (headless, non-interactive).",
    )

    # Backward compatibility modes
    parser.add_argument("--legacy-footfall", action="store_true", help="Run legacy CustomerCounter service.")
    parser.add_argument("--legacy-bag", action="store_true", help="Run legacy PersonBagCounter service.")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # Determine execution mode:
    # Service mode runs completely headless and unattended
    if args.service:
        args.no_display = True
    elif args.display:
        args.no_display = False

    is_interactive_env = sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False
    is_first_run = not os.path.exists("config/gates.json")

    # Prompt wizard if explicitly requested, or if run in interactive terminal without --service / --no-display
    should_prompt = args.interactive or (
        is_interactive_env
        and not args.service
        and not args.no_display
    )

    if should_prompt:
        print("=" * 65)
        print("    Retail AI Analytics — Edge Video Tracking & Demographics")
        if is_first_run:
            print("    [FIRST-TIME SETUP: Virtual Gates Not Configured]")
        print("=" * 65)
        print(f"\n[Current Source]: {mask_url(args.source)}")
        rtsp_input = input("\nEnter NVR RTSP Stream URL or Video Path\n(Press ENTER to keep current): ").strip()
        if rtsp_input:
            args.source = rtsp_input
            update_rtsp_url_in_config(args.config, rtsp_input)

        calib_default = "Y/n" if is_first_run else "y/N"
        calib_input = input(f"\nDo you want to calibrate virtual gates for this camera? [{calib_default}]: ").strip().lower()
        if is_first_run and calib_input in ("", "y", "yes"):
            args.calibrate = True
        elif not is_first_run and calib_input in ("y", "yes"):
            args.calibrate = True

        display_input = input("\nShow live video preview window? [Y/n]: ").strip().lower()
        if display_input in ("n", "no"):
            args.no_display = True
        else:
            args.no_display = False
        print("=" * 65)

    # 1. Optional pre-flight gate calibration
    if args.calibrate:
        print("[App] Launching Gate Calibration Tool...")
        try:
            from calibrate_gates import GateCalibrator

            calibrator = GateCalibrator(args.source)
            calibrator.run()
            print("[App] Gate calibration finished.")
        except Exception as exc:
            print(f"[App] Calibration error: {exc}")

    # 2. Legacy fallback modes
    if args.legacy_footfall:
        print("[App] Running in Legacy Footfall mode...")
        from services.customer_counter import CustomerCounter

        counter = CustomerCounter(args.source)
        counter.run()
        return

    if args.legacy_bag:
        print("[App] Running in Legacy Person-Bag Counter mode...")
        from services.customer_bag_counter import PersonBagCounter

        counter = PersonBagCounter(args.source)
        counter.run()
        return

    # 3. Default: Unified Low-Resource Retail Analytics Pipeline
    print("=" * 65)
    print("  Starting Re-Engineered Unified Retail Analytics Engine")
    demo_mode = "Disabled" if args.no_demographics else "Enabled (Async MiVOLO CPU)"

    print(f" Source:            {mask_url(args.source)}")
    print(f" Primary Weights:   {args.weights}")
    print(f" Demographics:      {demo_mode}")
    print(f" Display Mode:      {'Headless' if args.no_display else 'Decoupled GUI Preview Window'}")
    print(f" Target Hardware:   Intel CPU (No GPU), 4 Cores, ~2GB RAM Budget")
    print("=" * 65)

    pipeline = UnifiedRetailPipeline(
        video_source=args.source,
        config_path=args.config,
        primary_model_path=args.weights,
        enable_demographics=not args.no_demographics,
        display=not args.no_display,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

