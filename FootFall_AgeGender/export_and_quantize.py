"""Export YOLOv8n to ONNX (CPU-friendly) and optionally quantize.

Usage (needs `pip install ultralytics onnx onnxslim` only on the build machine,
NOT on the edge PC):

    python export_and_quantize.py                 # FP32 480x480 -> models/yolov8n_480.onnx
    python export_and_quantize.py --imgsz 416
    python export_and_quantize.py --quantize      # also writes models/yolov8n_480_int8.onnx

Then compare with benchmark.py. Dynamic INT8 is NOT always faster for conv
models on CPU, so keep whichever is faster/more accurate on YOUR hardware.
"""
import argparse
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="yolov8n.pt")
ap.add_argument("--imgsz", type=int, default=480)
ap.add_argument("--quantize", action="store_true")
args = ap.parse_args()

Path("models").mkdir(exist_ok=True)

from ultralytics import YOLO

exported = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12, dynamic=False)
fp32_path = Path("models") / f"yolov8n_{args.imgsz}.onnx"
shutil.copy(exported, fp32_path)
print(f"FP32 model: {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

if args.quantize:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = Path("models") / f"yolov8n_{args.imgsz}_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    print(f"INT8 model: {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
