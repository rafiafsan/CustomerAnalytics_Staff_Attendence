"""Benchmark ONNX / OpenVINO detector latency and RAM on this machine.

    python benchmark.py models/yolov8m.onnx
    python benchmark.py models/yolov8m.onnx --runtime openvino --device GPU
    python benchmark.py models/yolov8m.onnx --compare
"""
import argparse
import time

import numpy as np
import psutil

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


def bench_ort(model_path: str, threads: int, iters: int):
    proc = psutil.Process()
    ram0 = proc.memory_info().rss / 1048576
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    inp = sess.get_inputs()[0]
    h, w = (inp.shape[2], inp.shape[3])
    x = np.random.rand(1, 3, h, w).astype(np.float32)

    for _ in range(5):
        sess.run(None, {inp.name: x})

    proc.cpu_percent(None)
    lat = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        lat.append((time.perf_counter() - t) * 1000)
    cpu = proc.cpu_percent(None)
    lat = np.array(lat)
    ram_mb = proc.memory_info().rss / 1048576 - ram0

    return {
        "name": f"ONNX Runtime CPU ({threads} th)",
        "w": w, "h": h, "load_s": load_s,
        "ram_mb": ram_mb, "mean_ms": lat.mean(),
        "p95_ms": np.percentile(lat, 95), "fps": 1000 / lat.mean(),
        "cpu": cpu
    }


def bench_ov(model_path: str, device: str, threads: int, iters: int):
    proc = psutil.Process()
    ram0 = proc.memory_info().rss / 1048576
    core = ov.Core()
    if device == "CPU":
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": threads})

    t0 = time.perf_counter()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, device)
    load_s = time.perf_counter() - t0

    req = compiled.create_infer_request()
    inp = model.inputs[0]
    h, w = int(inp.shape[2]), int(inp.shape[3])
    x = np.random.rand(1, 3, h, w).astype(np.float32)

    for _ in range(5):
        req.infer({0: x})

    proc.cpu_percent(None)
    lat = []
    for _ in range(iters):
        t = time.perf_counter()
        req.infer({0: x})
        lat.append((time.perf_counter() - t) * 1000)
    cpu = proc.cpu_percent(None)
    lat = np.array(lat)
    ram_mb = proc.memory_info().rss / 1048576 - ram0

    return {
        "name": f"OpenVINO {device}",
        "w": w, "h": h, "load_s": load_s,
        "ram_mb": ram_mb, "mean_ms": lat.mean(),
        "p95_ms": np.percentile(lat, 95), "fps": 1000 / lat.mean(),
        "cpu": cpu
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to .onnx model")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--runtime", choices=["auto", "onnx", "openvino"], default="auto")
    ap.add_argument("--device", choices=["CPU", "GPU", "AUTO"], default=None)
    ap.add_argument("--compare", action="store_true", help="Run full comparison across all available runtimes")
    args = ap.parse_args()

    results = []
    if args.compare or args.runtime == "auto":
        if ONNX_AVAILABLE:
            print("[Benchmark] Testing ONNX Runtime CPU...")
            results.append(bench_ort(args.model, args.threads, args.iters))
        if OPENVINO_AVAILABLE:
            core = ov.Core()
            if "CPU" in core.available_devices:
                print("[Benchmark] Testing OpenVINO CPU...")
                results.append(bench_ov(args.model, "CPU", args.threads, args.iters))
            if "GPU" in core.available_devices:
                print("[Benchmark] Testing OpenVINO GPU (Intel iGPU)...")
                results.append(bench_ov(args.model, "GPU", args.threads, args.iters))
    elif args.runtime == "onnx":
        results.append(bench_ort(args.model, args.threads, args.iters))
    elif args.runtime == "openvino":
        dev = args.device or "CPU"
        results.append(bench_ov(args.model, dev, args.threads, args.iters))

    print("\n" + "=" * 80)
    print(f"BENCHMARK REPORT: {args.model}")
    print("=" * 80)
    print(f"{'Runtime':<28} | {'Load':<7} | {'Mean Latency':<12} | {'P95 Latency':<11} | {'Throughput':<9}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<28} | {r['load_s']:>5.2f}s | {r['mean_ms']:>8.1f} ms | {r['p95_ms']:>7.1f} ms | {r['fps']:>6.1f} FPS")
    print("=" * 80)


if __name__ == "__main__":
    main()
