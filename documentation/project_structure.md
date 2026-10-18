# Project Structure & Component Breakdown
==========================================

This document details the complete hierarchical directory layout of the **OmniRetail AI Suite** workspace and the functional role of each file and folder.

---

## 🌳 Hierarchical Directory Tree

```
CustomerAnalytics_Staff_Attendence/
│
├── run_unified_app.bat                  # 1-Click launcher setting RTSP/CPU flags and starting unified_app.py
├── unified_app.py                      # Master Python launcher (resolves imports and initializes GUI)
├── unified_gui.py                      # Master native Windows Desktop GUI (Tkinter/TTK)
├── requirements.txt                    # Project-wide Python dependencies
├── README.md                           # Main project setup and operations guide
├── README_UNIFIED.md                   # Legacy technical overview of unified desktop app
├── edge_events.db                      # Local SQLite database for root edge events
│
├── documentation/                      # Dedicated Technical Documentation Hub
│   ├── README.md                       # Documentation table of contents and overview
│   ├── project_structure.md            # Detailed structural breakdown (this file)
│   ├── architecture_and_diagrams.md    # Mermaid architecture and workflow diagrams
│   ├── pipeline_and_api_spec.md        # REST API contracts, payloads, and SQLite schemas
│   └── deployment_and_exe_guide.md     # Production PyInstaller build and service guide
│
├── Attendence System/                  # VisionGuard Staff Facial Recognition Engine
│   ├── pipeline.py                     # 8-Stage Vision Pipeline (Detection, Quality, SFace, Temporal, API)
│   ├── detector.py                     # YuNet Face & 5-Landmark Detector (OpenCV DNN ONNX)
│   ├── quality.py                      # Face Quality Assessor (blur/sharpness, yaw, roll, size)
│   ├── recognizer.py                   # SFace 128-d Embedding Extractor & Cosine Distance Matcher
│   ├── temporal.py                     # Multi-frame confirmation streak tracker & cooldown logic
│   ├── stream.py                       # Thread-safe RTSP stream grabber with FPS throttling
│   ├── database.py                     # SQLite interface for employee profiles, embeddings, and punch logs
│   ├── storage.py                      # File system manager for aligned 112x112 crops and .npy embeddings
│   ├── config.py                       # VisionGuard configuration, thresholds, and paths
│   ├── enroll_employee.py              # CLI utility to register staff from a portrait photo
│   ├── enroll_staffs.py                # Batch staff enrollment helper script
│   ├── verify_embeddings.py            # Utility to test and verify gallery embedding integrity
│   ├── view_attendance.py              # Command-line inspection tool for SQLite attendance logs
│   ├── check_verified.py               # Quick validation script for confirmed staff punch records
│   ├── compare_live_with_staff.py      # Diagnostic script comparing live camera feed to enrolled staff
│   ├── web_app.py                      # Optional standalone FastAPI web application for attendance
│   ├── main_gui.py                     # Standalone Attendance GUI launcher
│   ├── requirements.txt                # Attendance-specific dependencies
│   ├── Staffs/                         # Directory containing registered staff photos organized by ID_Name
│   ├── models/                         # Face detection and recognition weights
│   │   ├── face_detection_yunet_2023mar.onnx   # YuNet face detector model
│   │   └── face_recognition_sface_2021dec.onnx # SFace 128-dimensional embedding model
│   ├── data/
│   │   └── attendance.db               # SQLite database: employees, embeddings, punches, capture audit
│   ├── captures/                       # Storage for processed aligned chips and raw face crops
│   ├── static/                         # Static CSS/JS assets for the optional web portal
│   └── templates/                      # HTML templates for the optional web portal
│
├── FootFall_AgeGender/                 # Customer Analytics & Virtual Gate Engine
│   ├── app.py                          # Headless CLI service for FootFall and Demographics
│   ├── calibrate_gates.py              # Interactive OpenCV Virtual Gate calibration utility
│   ├── main_gui.py                     # Standalone Customer Analytics GUI
│   ├── benchmark.py                    # Performance benchmarking tool for inference FPS
│   ├── yolo_dataset_auditor.py         # Diagnostic dataset auditor tool
│   ├── export_and_quantize.py          # Script for exporting/quantizing YOLO ONNX models
│   ├── requirements.txt                # Footfall edge runtime dependencies
│   ├── edge_events.db                  # Local SQLite store for footfall crossing and dwell events
│   ├── camera/
│   │   └── rtsp_reader.py              # Decoupled camera frame reader (OpenCV/FFmpeg)
│   ├── config/
│   │   ├── production.yaml             # Production settings (camera URL, shop ID, thresholds)
│   │   ├── settings.py                 # Python settings constants and config loader
│   │   └── gates.json                  # Virtual gate polygon coordinates (Entry, Exit, Dwell, Bag)
│   ├── database/
│   │   ├── db_manager.py               # SQLite/PostgreSQL manager for edge events and occupancy
│   │   └── schema.sql                  # Database table definitions
│   ├── detectors/
│   │   ├── person_bag_associator.py    # Spatial associator linking shoppers with carried bags
│   │   └── person_bag_exit_validator.py# Validator confirming bag crossing during store exit
│   ├── gate/
│   │   ├── geometry.py                 # Geometric polygon intersection and line crossing math
│   │   ├── event_manager.py            # Event generator for valid gate transitions
│   │   └── unified_gate_manager.py     # Coordinates multi-gate state (Entry, Exit, Dwell)
│   ├── inference/
│   │   ├── detector.py                 # YOLOv8 ONNX/OpenVINO inference wrapper
│   │   └── inference_scheduler.py      # Frame scheduler regulating CPU inference load
│   ├── mivolo_system/                  # MiVOLO Demographic Profiler
│   │   ├── mivolo/                     # MiVOLO deep learning architecture (timm backbones)
│   │   └── predictor.py                # Wrapper predicting age (years) and gender (Male/Female)
│   ├── monitoring/
│   │   └── performance_monitor.py      # Real-time FPS, RAM, and CPU telemetry collector
│   ├── services/
│   │   ├── unified_pipeline.py         # Main footfall & demographics orchestration engine
│   │   └── business_logic.py           # Dwell time validation, entry/exit business rules
│   ├── tracking/
│   │   └── track_manager.py            # ByteTrack object tracker maintaining identity across frames
│   ├── utils/
│   │   ├── security.py                 # Password & URL masking utilities
│   │   └── system_monitor.py           # Lightweight OS resource probe
│   ├── visualization/
│   │   └── draw_track.py               # OpenCV visual renderer for bounding boxes, gates, and trails
│   ├── models/                         # Object detection and demographic models
│   │   ├── yolov8m.onnx                # YOLOv8 Medium ONNX detection model (320x320)
│   │   ├── yolov8m_int8.onnx           # Quantized INT8 model for ultra-low CPU utilization
│   │   └── model_imdb_cross_person_4.22_99.46.pth.tar # MiVOLO demographic weights
│   └── test_videos/                    # Sample outlet camera recordings for offline testing
│
└── models/                             # Shared root model weights directory
```

---

## 🔍 Detailed Component Descriptions

### 1. Unified Master Layer
* **`run_unified_app.bat`**: Sets environment variables (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`, `CUDA_VISIBLE_DEVICES=-1`) and launches `unified_app.py`.
* **`unified_app.py`**: Solves Python module import collisions between `FootFall_AgeGender` and `Attendence System` using dynamic module bridging (`importlib`), then hands control to `unified_gui.py`.
* **`unified_gui.py`**: Native Windows desktop interface built with Tkinter. Manages 4 tabs:
  1. Store Executive Dashboard (dual live feeds, summary metrics, quick actions)
  2. Customer Analytics (virtual gates canvas, footfall event log)
  3. VisionGuard Staff Attendance (live face badges, roster, 3-photo registration modal)
  4. Central Settings & Diagnostics (camera URLs, shop ID, 1-click self-test diagnostic tool)

---

### 2. VisionGuard Staff Attendance Engine (`Attendence System/`)
* **`pipeline.py`**: Executes the 8-stage face recognition pipeline on every sampled frame. Contains `AttendanceApiClient` which dispatches confirmed punch events asynchronously to the cloud REST API.
* **`detector.py`**: Wraps the **YuNet ONNX** model. Extracts bounding boxes and 5 facial landmark points (eyes, nose, mouth corners) in ~8–15 ms on CPU.
* **`quality.py`**: Assesses facial quality before recognition. Rejects faces that are blurry (Laplacian variance), severely angled (yaw/roll ratio), or too small.
* **`recognizer.py`**: Wraps **SFace ONNX**. Aligns the face into a normalized $112 \times 112$ crop using affine transformation, then extracts a 128-dimensional L2-normalized embedding. Matches embeddings using cosine similarity.
* **`temporal.py`**: Implements streak-based confirmation. Requires the same identity to be matched in multiple consecutive frames within a sliding time window before confirming attendance, preventing false positives from fleeting visitors.
* **`database.py`**: SQLite database manager (`attendance.db`). Manages tables for employees, multi-view embeddings, verified punches, and quality failure audit trails.
* **`storage.py`**: Saves aligned face images and `.npy` embedding vectors to disk.

---

### 3. Customer Traffic Analytics Engine (`FootFall_AgeGender/`)
* **`services/unified_pipeline.py`**: Master orchestration loop for footfall tracking. Coordinates frame reading, YOLO detection, ByteTrack tracking, gate crossing math, and demographic profiling.
* **`inference/detector.py`**: Wraps **YOLOv8 ONNX** (resized to $320 \times 320$) running via ONNX Runtime with CPU execution providers. Detects persons and bags.
* **`tracking/track_manager.py`**: Multi-object tracker using the **ByteTrack** algorithm. Associates detections across frames by matching Kalman filter predictions with Hungarian assignment.
* **`gate/unified_gate_manager.py` & `geometry.py`**: Tracks the trajectory of each tracked person ID across defined virtual gate polygons (Entry, Exit, Dwell). Computes line-crossing directionality and enforces anti-bounce dwell time (e.g., must stay inside 60 seconds to count as valid entry).
* **`mivolo_system/`**: Deep learning framework based on a lightweight Vision Transformer / ConvNeXt backbone for estimating customer age and gender from the upper body / face crop.
* **`calibrate_gates.py`**: Interactive calibration tool where users can click on a frozen camera frame to redraw virtual gate boundaries and save them to `gates.json`.
