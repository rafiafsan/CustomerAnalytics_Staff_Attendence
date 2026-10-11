# VisionGuard AI — Real-Time Face Attendance & Verification System

A production-grade, 8-stage facial recognition and attendance monitoring pipeline using **OpenCV YuNet**, **SFace 128-d embeddings**, **RTSP low-latency capture**, and **SQLite audit storage**.

---

## 8-Stage Architecture Pipeline

```
  [1] RTSP Camera Stream (HEVC / 1080p @ 10.1.56.11:554)
             │
             ▼
  [2] Low-Latency Frame Sampling (2–5 FPS Zero-Buffer Drift)
             │
             ▼
  [3] YuNet Face & Landmark Detection (5 Facial Landmarks)
             │
             ▼
  [4] Quality Filter & Pose Alignment
      ├── Size Filter (Min 45x45px, Aspect Ratio)
      ├── Blur Detection (Laplacian Variance >= 65.0)
      ├── Orientation Checks (Roll <= 30°, Yaw Symmetry >= 0.35)
      └── 112x112 Affine Landmark Alignment
             │
             ▼
  [5] SFace Embedding Extraction (128-dimensional L2-normalized vector)
             │
             ▼
  [6] Identity Matching (Cosine Distance against Enrolled Gallery)
             │
             ▼
  [7] Temporal Multi-Frame Confirmation (3 matches within 4s window + 5m cooldown)
             │
             ▼
  [8] Local Storage & SQLite Audit
      ├── Aligned Face Crops (.jpg in `data/faces/`)
      ├── 128-d Feature Embeddings (.npy & .json in `data/embeddings/`)
      └── SQLite Relational Database (`data/attendance.db`)
```

---

## Hardware & Network Configuration
* **RTSP URL**: `rtsp://ctrl:ctl%40123456@10.1.56.11:554/Streaming/Channels/1201`
* **Transport**: TCP (`rtsp_transport;tcp`)
* **Camera Resolution**: 1920×1080 HEVC

---

### 1. Batch-Enroll All Staff from `Staffs/` Folder
Extracts high-quality 128-d SFace embeddings for all photos in `Staffs/Staff_1` through `Staffs/Staff_5`, generates multi-angle templates + centroid mean embeddings, and registers them in SQLite:

```bash
python enroll_staffs.py
```

### 2. Run Live RTSP Capture & Recognition Demo (Console)
Connects to the RTSP camera, samples at 3 FPS, performs face detection, quality filtering, SFace embedding extraction, and matches against all enrolled staff:

```bash
python run_capture_demo.py --seconds 30 --fps 3.0
```

### 2. Verify Face Embeddings on Local Drive
Compare any two saved `.npy` / `.json` embeddings to verify if they belong to the same person:

```bash
# List all captured embeddings with quality scores & IDs:
python verify_embeddings.py --list

# Compare two embeddings:
python verify_embeddings.py --file1 data/embeddings/face_20261010_120000_emb.npy --file2 data/embeddings/face_20261010_120005_emb.npy
```

### 3. Enroll Employees
Enroll an employee from a portrait photo or directly from the live camera stream:

```bash
# Enroll from a photo:
python enroll_employee.py --id EMP-001 --name "Sarah Connor" --dept "Security" --image sample.jpg

# Enroll from live RTSP camera:
python enroll_employee.py --id EMP-002 --name "John Doe" --dept "Engineering"
```

### 4. Launch Windows Native Desktop GUI (Recommended)
Start the complete native Windows GUI application (zero browser / web server required). Features live RTSP stream, multi-angle 3-photo staff registration dialog, live staff roster with confirmation status, interactive embedding tester, and exportable data tables:

```bash
# Direct launcher (or double-click run_gui.bat):
python main_gui.py
```

* **Live RTSP Stream Viewport**: Real-time bounding boxes, 5 YuNet facial landmark dots, recognition badges, and snapshot capture.
* **Top Metric KPI Cards**: Frame sampling rate, total captures, quality filter pass rate, enrolled staff count, and attendance confirmed today.
* **Staff Roster & Real-Time Attendance**: Searchable staff list with avatars, punch timestamps, punch count, and delete actions.
* **Register Staff (3 Photos)**: Multi-angle registration modal saving to `Staffs/{staff_id}_{name}` with centroid template extraction.
* **Embedding Verification Tester**: Select Reference and Probe embeddings to compute Cosine Similarity, L2 distance, and match verdicts.
* **Data Explorer Tabs**: Stored face crops gallery, verified attendance history table (CSV exportable), quality audit logs, and live terminal console.

### 5. Launch Modern Web Dashboard (Alternative)
If browser-based remote access is still needed:

```bash
python web_app.py
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser.

---

## File Structure

```
Attendence System/
├── config.py             # Global configuration (URL, thresholds, paths)
├── stream.py             # Threaded RTSP capture (zero latency, auto-reconnect)
├── detector.py           # OpenCV YuNet FaceDetectorYN wrapper
├── quality.py            # Quality assessor (blur, size, roll, yaw, borders)
├── recognizer.py         # OpenCV SFace FaceRecognizerSF (112x112 align, 128-d feat)
├── temporal.py           # Multi-frame match tracker & cooldown logic
├── database.py           # SQLite database schema and logging
├── storage.py            # Local disk storage for embeddings (.npy/.json) & faces
├── pipeline.py           # Integrated 8-stage orchestrator
├── run_capture_demo.py   # Main standalone capture demo script
├── verify_embeddings.py  # Face embedding verification & comparison tool
├── enroll_employee.py    # Employee enrollment utility
├── demo_gui.py           # OpenCV GUI live stream window
├── web_app.py            # FastAPI web server
├── models/               # ONNX models (YuNet 2023mar & SFace 2021dec)
├── data/
│   ├── faces/            # Aligned 112x112 face chips (.jpg)
│   ├── embeddings/       # 128-d vector embeddings (.npy & .json)
│   └── attendance.db     # SQLite database
├── templates/            # Web dashboard HTML
└── static/               # CSS and JavaScript
```
