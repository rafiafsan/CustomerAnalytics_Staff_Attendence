# How to Run the Retail AI Analytics System (Low-Resource PC Guide)

This guide walks you through setting up, generating the model, configuring, and running the optimized computer vision system on your PC.

---

## 1. Quick Requirements Overview

* **Operating System:** Windows 10/11 (64-bit) or Ubuntu Linux
* **Hardware:** Intel Celeron / Pentium / Core i3/i5 (4 Cores, 2–4 GB Usable RAM, No GPU required)
* **Python Version:** Python 3.10, 3.11, or 3.12 (64-bit)

---

## 2. Step 1: Install Dependencies

Open PowerShell or Command Prompt inside the project directory:

```powershell
# 1. (Recommended) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install edge runtime dependencies
pip install -r requirements.txt
```

> **Note:** If you also want to generate/convert the YOLO model on this machine, install the export tools:
> ```powershell
> pip install -r requirements-dev.txt
> ```

---

## 3. Step 2: Prepare the YOLO ONNX Model

The system uses an optimized **YOLOv8n ONNX** model (~6 MB) for high-speed CPU detection of both persons and bags.

Run the model generator:

```powershell
python export_and_quantize.py --weights yolov8n.pt --imgsz 480
```

* This automatically downloads `yolov8n.pt`, exports it to ONNX at 480×480 resolution, and places it inside `models/yolov8n_480.onnx`.
* *(Optional)* To create an INT8 quantized version:
  ```powershell
  python export_and_quantize.py --weights yolov8n.pt --imgsz 480 --quantize
  ```

---

## 4. Step 3: Configure Settings

Check `config/production.yaml`:

```yaml
camera:
  fallback_video: "test_videos/Outlet_Sample_1_Banasree.avi"
  width: 1280
  height: 720
  target_fps: 15

inference:
  enabled: true
  target_fps: 4
  model: "models/yolov8n_480.onnx"
  backend: "onnx"
  cpu_threads: 2
  confidence: 0.40

api:
  enabled: true
  base_url: "http://ai.prangroup.com:8021"
  api_key: ""  # Set via environment variable API_KEY
```

### Setting API Key / Secrets (Optional)
In PowerShell:
```powershell
$env:API_KEY="your_secret_token"
```
Or in CMD / Batch (`run_service.bat` automatically looks for `edge.env.bat`).

---

## 5. Step 4: Run the Application

### Option A: Standard Run with Preview Window (GUI Mode)

Run on default video:
```powershell
python app.py
```

Run on a specific video file:
```powershell
python app.py --source "test_videos/Outlet_Sample_1_Banasree.avi"
```

Run on a live RTSP CCTV camera stream:
```powershell
python app.py --source "rtsp://admin:password123@192.168.1.100:554/Streaming/Channels/102"
```

Run without demographics (saves ~300 MB RAM and CPU cycles):
```powershell
python app.py --no-demographics
```

---

### Option B: High-Performance Headless Mode (Recommended for Edge PC)

Runs in the background without opening an OpenCV GUI window (saves RAM and GPU/display CPU):

```powershell
python app.py --no-display --no-demographics
```

---

### Option C: Production 24/7 Auto-Restart Service

Double-click `run_service.bat` or run:

```powershell
.\run_service.bat
```

* Automatically restarts the application in 5 seconds if it exits or crashes.
* Logs are written and automatically rotated in `retail_ai.log`.
* Events are buffered safely into local SQLite `edge_events.db` if the network goes down.

---

## 6. Virtual Gate Calibration (Optional)

If your camera angle changes or you want to redraw entry/exit tripwire gates:

```powershell
python calibrate_gates.py --source "test_videos/Outlet_Sample_1_Banasree.avi"
```

* Click two points on screen to set the gate line.
* Click a third point inside the store as the inside reference direction.
* Press `s` to save to `config/gates.json`.

---

## 7. Keyboard Shortcuts (During Live Video Preview)

| Key | Action |
| :---: | :--- |
| **`q`** or **`ESC`** | Safely quit the application and release camera resources |
| **`r`** | Hot-reload gate coordinates from `config/gates.json` without stopping |

---

## 8. Benchmark Your PC Performance

To verify model latency and CPU usage on your PC:

```powershell
python benchmark.py models/yolov8n_480.onnx --threads 2
```
