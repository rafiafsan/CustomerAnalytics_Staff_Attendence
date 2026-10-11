"""Detector Factory for Edge Inference
====================================
Instantiates either ONNXDetector or TorchDetector based on configuration.
Ensures single-instance model lifecycle.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

from pathlib import Path
from .onnx_detector import ONNXDetector, ONNX_AVAILABLE, OPENVINO_AVAILABLE


def _find_model_file(model_path: str) -> Optional[str]:
    """Find model path across workspace subfolders and parent directories."""
    if not model_path:
        return None
    p = Path(model_path)
    if p.is_file():
        return str(p.resolve())

    this_dir = Path(__file__).resolve().parent
    ff_root = this_dir.parent
    repo_root = ff_root.parent

    search_dirs = [
        Path.cwd(),
        Path.cwd() / "models",
        Path.cwd() / "FootFall_AgeGender",
        Path.cwd() / "FootFall_AgeGender" / "models",
        ff_root,
        ff_root / "models",
        repo_root,
        repo_root / "models",
        repo_root / "FootFall_AgeGender",
        repo_root / "FootFall_AgeGender" / "models",
    ]

    for base in search_dirs:
        cand = (base / p.name).resolve()
        if cand.is_file():
            return str(cand)
        cand_rel = (base / p).resolve()
        if cand_rel.is_file():
            return str(cand_rel)

    # Secondary fallback to known ONNX model names
    for cand_name in ("yolov8m.onnx", "yolov8m_int8.onnx", "yolov8n.onnx"):
        for base in search_dirs:
            cand = (base / cand_name).resolve()
            if cand.is_file():
                return str(cand)
            cand_m = (base / "models" / cand_name).resolve()
            if cand_m.is_file():
                return str(cand_m)

    return None


def create_detector(config: Dict[str, Any]):
    """Instantiate the configured primary detector."""
    inf_cfg = config.get("inference", {})
    raw_model_path = inf_cfg.get("model", "models/yolov8m.onnx")
    fallback_model = inf_cfg.get("fallback_model", "models/yolov8n.pt")
    conf_thresh = float(inf_cfg.get("confidence", 0.40))
    cpu_threads = int(inf_cfg.get("cpu_threads", 2))
    backend = inf_cfg.get("backend", "onnx").lower()
    device = inf_cfg.get("device", None)
    prefer_openvino = inf_cfg.get("prefer_openvino", True)

    # Robust path resolution
    resolved_model = _find_model_file(raw_model_path)
    if resolved_model:
        model_path = resolved_model
    else:
        model_path = raw_model_path

    # Determine if ONNX/OpenVINO can be used
    if backend in ("onnx", "openvino") and model_path.endswith(".onnx") and os.path.exists(model_path) and (ONNX_AVAILABLE or OPENVINO_AVAILABLE):
        try:
            return ONNXDetector(
                model_path=model_path,
                conf_threshold=conf_thresh,
                cpu_threads=cpu_threads,
                prefer_openvino=prefer_openvino,
                device=device,
            )
        except Exception as e:
            print(f"[create_detector] Warning: Failed to initialize ONNX/OpenVINO detector ({e}). Falling back to Torch...")

    # PyTorch Fallback
    target_pt = model_path if model_path.endswith(".pt") and os.path.exists(model_path) else _find_model_file(fallback_model)
    if not target_pt or not os.path.exists(target_pt):
        # Last resort check
        for candidate in ("models/yolov8n.pt", "models/yolov8m.pt", "models/yolov8l.pt"):
            found = _find_model_file(candidate)
            if found and os.path.exists(found):
                target_pt = found
                break

    try:
        from .torch_detector import TorchDetector  # lazy: pulls in torch + ultralytics
        imgsz = int(inf_cfg.get("imgsz", 384))
        return TorchDetector(
            model_path=target_pt,
            conf_threshold=conf_thresh,
            cpu_threads=cpu_threads,
            imgsz=imgsz,
        )
    except Exception as e:
        raise RuntimeError(
            f"Neither ONNX runtime nor PyTorch could load detector. "
            f"ONNX candidate: {model_path} (exists={os.path.exists(model_path)}). "
            f"PyTorch error: {e}"
        )

