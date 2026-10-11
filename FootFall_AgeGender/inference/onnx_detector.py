"""Lightweight ONNX Runtime CPU YOLO Detector
=============================================
Optimized for low-resource Intel CPU machines (No GPU, 4 CPU cores, ~2GB RAM).
Runs YOLOv8 ONNX models using ONNX Runtime with strictly capped CPU threads.
Performs fast C++ accelerated NMS via cv2.dnn.NMSBoxes.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXDetector:
    """High-performance YOLOv8 detector supporting OpenVINO and ONNX Runtime.

    Detects 'person' (COCO class 0) and bag classes (backpack=24, handbag=26, suitcase=28).
    Automatically runs the model using OpenVINO Runtime (Intel iGPU / CPU) when available,
    with seamless fallback to ONNX Runtime CPU.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        cpu_threads: int = 2,
        prefer_openvino: bool = True,
        device: Optional[str] = None,
    ):
        if not OPENVINO_AVAILABLE and not ONNX_AVAILABLE:
            raise RuntimeError("Either openvino or onnxruntime is required. Run: pip install openvino onnxruntime")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cpu_threads = cpu_threads
        self.device = device
        self.backend_name = "onnxruntime"

        self._target_cids = np.array([0, 24, 26, 28], dtype=np.int32)
        # Target class IDs (COCO): 0 = person, 24 = backpack, 26 = handbag, 28 = suitcase
        self.target_classes = {0: "person", 24: "bag", 26: "bag", 28: "bag"}

        t0 = time.time()

        # 1. Prefer OpenVINO runtime if available
        if prefer_openvino and OPENVINO_AVAILABLE:
            try:
                core = ov.Core()
                # Enable model caching to load compiled GPU/CPU model in ~1s on subsequent startups
                cache_dir = os.path.join(os.path.dirname(model_path) or ".", ".openvino_cache")
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    core.set_property({"CACHE_DIR": cache_dir})
                except Exception:
                    pass

                available = core.available_devices
                chosen_device = self.device.upper() if self.device else ("GPU" if "GPU" in available else "CPU")
                if chosen_device == "CPU":
                    core.set_property("CPU", {"INFERENCE_NUM_THREADS": self.cpu_threads})

                print(f"[ONNXDetector] Loading model {model_path} via OpenVINO Runtime on {chosen_device}...")
                ov_model = core.read_model(model_path)
                self.ov_compiled = core.compile_model(ov_model, chosen_device)
                self.ov_infer_request = self.ov_compiled.create_infer_request()
                self.backend_name = f"openvino_{chosen_device.lower()}"

                inp_shape = ov_model.inputs[0].shape
                self.input_height = int(inp_shape[2]) if len(inp_shape) >= 4 else 320
                self.input_width = int(inp_shape[3]) if len(inp_shape) >= 4 else 320
                self.load_time_sec = time.time() - t0

                print(
                    f"[ONNXDetector] Model loaded in {self.load_time_sec:.2f}s via OpenVINO ({chosen_device})! "
                    f"Input size: {self.input_width}x{self.input_height}"
                )
                return
            except Exception as e:
                print(f"[ONNXDetector] OpenVINO init failed ({e}), falling back to ONNX Runtime CPU...")

        # 2. Fallback to ONNX Runtime CPU
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is required for ONNX fallback. Run: pip install onnxruntime")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.cpu_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"[ONNXDetector] Loading model {model_path} with ONNX Runtime ({self.cpu_threads} CPU threads)...")
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.load_time_sec = time.time() - t0

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # e.g. [1, 3, 320, 320] or [1, 3, 640, 640]
        self.input_height = self.input_shape[2] if len(self.input_shape) >= 4 and isinstance(self.input_shape[2], int) else 320
        self.input_width = self.input_shape[3] if len(self.input_shape) >= 4 and isinstance(self.input_shape[3], int) else 320
        self.output_name = self.session.get_outputs()[0].name
        self.backend_name = "onnxruntime"

        print(
            f"[ONNXDetector] Model loaded in {self.load_time_sec:.2f}s via ONNX Runtime! "
            f"Input size: {self.input_width}x{self.input_height}"
        )

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Letterbox BGR frame into a normalized NCHW float32 tensor.

        Returns (tensor, ratio, pad_x, pad_y) so model coords can be mapped back.
        """
        h, w = frame.shape[:2]
        ratio = min(self.input_width / w, self.input_height / h)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        if (new_w, new_h) != (w, h):
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        pad_x = (self.input_width - new_w) / 2.0
        pad_y = (self.input_height - new_h) / 2.0
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        if canvas.shape[0] != self.input_height or canvas.shape[1] != self.input_width:
            canvas = cv2.resize(canvas, (self.input_width, self.input_height))

        # BGR -> RGB, HWC -> CHW, normalize to 0..1 in one pass
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
        tensor *= (1.0 / 255.0)
        return tensor[np.newaxis], ratio, float(left), float(top)

    def detect(
        self, frame: np.ndarray, original_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[Dict], List[Dict], float]:
        """Execute detection pass and return (person_detections, bag_detections, latency_ms).

        Detections are scaled to the original frame coordinate space.
        """
        t0 = time.time()
        fh, fw = frame.shape[:2]
        orig_h, orig_w = original_shape if original_shape else (fh, fw)

        # 1. Preprocess (letterbox)
        tensor, ratio, pad_x, pad_y = self.preprocess(frame)

        # model space -> inference-frame space -> original frame space
        sx = orig_w / float(fw)
        sy = orig_h / float(fh)

        # 2. Inference
        if self.backend_name.startswith("openvino"):
            outputs = self.ov_infer_request.infer({0: tensor})
            preds = outputs[0]
        else:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            preds = outputs[0]

        # Output shape: [1, 84, N] -> transpose to [N, 84]
        preds = np.squeeze(preds, axis=0) if preds.ndim == 3 else preds
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # Shape: (N, 84)

        boxes_list = []
        confidences = []
        class_ids = []

        # Coordinate columns: 0,1,2,3 = cx, cy, w, h ; class scores: 4..83
        boxes_xywh = preds[:, :4]
        # Only consider person/bag classes so unrelated COCO classes cannot mask them
        target_scores = preds[:, 4 + self._target_cids]
        best_idx = np.argmax(target_scores, axis=1)
        best_scores = target_scores[np.arange(target_scores.shape[0]), best_idx]
        valid_mask = best_scores >= self.conf_threshold

        valid_boxes = boxes_xywh[valid_mask]
        valid_scores = best_scores[valid_mask]
        valid_classes = self._target_cids[best_idx[valid_mask]]

        for box, score, cid in zip(valid_boxes, valid_scores, valid_classes):
            cx, cy, bw, bh = box
            x1 = ((cx - bw / 2.0) - pad_x) / ratio * sx
            y1 = ((cy - bh / 2.0) - pad_y) / ratio * sy
            w_box = (bw / ratio) * sx
            h_box = (bh / ratio) * sy
            boxes_list.append([int(x1), int(y1), int(w_box), int(h_box)])
            confidences.append(float(score))
            class_ids.append(int(cid))

        # 3. Fast C++ NMS
        person_detections: List[Dict] = []
        bag_detections: List[Dict] = []

        if boxes_list:
            indices = cv2.dnn.NMSBoxes(boxes_list, confidences, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, w_box, h_box = boxes_list[idx]
                    cid = class_ids[idx]
                    score = confidences[idx]
                    x1 = max(0, min(orig_w - 1, x))
                    y1 = max(0, min(orig_h - 1, y))
                    x2 = max(0, min(orig_w - 1, x + w_box))
                    y2 = max(0, min(orig_h - 1, y + h_box))

                    bbox = (x1, y1, x2, y2)
                    cname = self.target_classes[cid]

                    if cname == "person":
                        upper_center = (int((x1 + x2) / 2), int(y1))
                        bottom_center = (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.75))
                        lower_body_midpoint = (int((x1 + x2) / 2), int(y2))
                        person_detections.append({
                            "bbox": bbox,
                            "confidence": score,
                            "class_id": 0,
                            "class_name": "person",
                            "center": upper_center,
                            "bottom_center": bottom_center,
                            "person_lower_body_midpoint": lower_body_midpoint,
                        })
                    else:
                        midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        bag_detections.append({
                            "bbox": bbox,
                            "confidence": score,
                            "class_id": 26,
                            "class_name": "bag",
                            "bag_midpoint": midpoint,
                        })

        latency_ms = (time.time() - t0) * 1000.0
        return person_detections, bag_detections, latency_ms
