# OmniRetail AI Suite — Unified Enterprise Desktop Gateway
### Customer Traffic Analytics, Virtual Gate Inflow/Outflow & VisionGuard Staff Attendance

[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-blue.svg)](https://microsoft.com)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-green.svg)](https://python.org)
[![Inference: CPU Only](https://img.shields.io/badge/Hardware-Edge%20Mini%20PC%20(CPU)-orange.svg)]()
[![API: REST Gateway](https://img.shields.io/badge/Sync-HTTP%20REST%20API-purple.svg)]()

---

## 📌 Executive Summary

**OmniRetail AI Suite** is an industrial-grade, edge-deployed desktop application designed for retail outlets and supershops. It runs on low-cost hardware (**Intel Mini PC, 4 CPU cores, ~2–4 GB RAM, No GPU**) and unifies two edge computer vision systems:

1. **Customer Traffic & Demographics Analytics (`FootFall_AgeGender`):**
   * Real-time bidirectional store entry/exit counting across configurable virtual gates.
   * Dwell validation (filters out passing shoppers, security guards, and loitering staff).
   * Event-driven bag exit validation.
   * Demographic profiling (age group and gender estimation via CPU-optimised MiVOLO).
   * Real-time store occupancy tracking.

2. **VisionGuard Staff Attendance System (`Attendence System`):**
   * 8-stage facial recognition pipeline: YuNet landmark detection, quality validation (blur, roll, yaw), SFace 128-d embeddings, and multi-frame temporal confirmation.
   * 3-Photo multi-angle staff registration with EXIF auto-orientation and centroid template calculation.
   * Local SQLite audit logging with instant non-blocking cloud reporting.

> [!IMPORTANT]
> **Cloud API Integration (No Direct HRIS/Oracle DB on Edge):**
> Attendance punches and footfall events are transmitted securely to the central enterprise database over HTTP via the **Retail AI REST API Gateway** (`https://ai.prangroup.com/api/v1/attendance`). The edge outlet PC never makes direct database connections to Oracle or PostgreSQL, ensuring firewall safety, network resilience, and zero requirement for heavy database client drivers.

---

## 📂 Project Architecture at a Glance

```
CustomerAnalytics_Staff_Attendence/
│
├── run_unified_app.bat          # 1-Click launcher for Unified Desktop GUI
├── unified_app.py              # Master application entrypoint (headless launcher)
├── unified_gui.py              # Complete native Dark Slate Desktop GUI (Tkinter/TTK)
├── requirements.txt            # Unified project dependencies
├── README.md                   # This project setup & operations manual
│
├── documentation/              # Dedicated Comprehensive Documentation Folder
│   ├── README.md               # Documentation Table of Contents
│   ├── project_structure.md    # Full directory tree & component breakdown
│   ├── architecture_and_diagrams.md # System & Pipeline Workflow Mermaid Diagrams
│   ├── pipeline_and_api_spec.md# REST API payload specs & SQLite schemas
│   └── deployment_and_exe_guide.md  # Production packaging & PyInstaller guide
│
├── FootFall_AgeGender/         # Customer Analytics & Virtual Gate Engine
│   ├── app.py                  # Standalone background footfall service
│   ├── calibrate_gates.py      # Interactive OpenCV Gate Calibrator
│   ├── config/                 # YAML settings & gate polygon coordinates
│   ├── detectors/              # YOLOv8 Person/Bag detectors & associators
│   ├── mivolo_system/          # Demographic profiling (Age & Gender)
│   ├── tracking/               # ByteTrack multi-object tracking engine
│   └── models/                 # YOLOv8 ONNX & PyTorch weights
│
└── Attendence System/          # VisionGuard Staff Attendance Engine
    ├── pipeline.py             # 8-Stage Face Recognition & Attendance Pipeline
    ├── detector.py             # YuNet Face & 5-Landmark Detector
    ├── quality.py              # Quality Assessor (blur score, yaw, roll angles)
    ├── recognizer.py           # SFace 128-d Feature Extractor & Cosine Matcher
    ├── temporal.py             # Multi-Frame Temporal Confirmation Tracker
    ├── database.py             # Local SQLite Attendance & Audit Trail Manager
    ├── storage.py              # Aligned Face Crop & Embedding File Manager
    ├── models/                 # YuNet ONNX & SFace ONNX models
    └── Staffs/                 # Enrolled staff photo directories & templates
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
* **Operating System:** Windows 10 / 11 (64-bit)
* **Python:** Version 3.10, 3.11, or 3.12 (64-bit)
* **Hardware:** Any modern Intel Core i3/i5 Mini PC with 4+ CPU cores, 4 GB RAM. No GPU required.

### 2. Environment Setup

Open PowerShell or Command Prompt in the repository folder:

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\activate

# 3. Upgrade pip and install all required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

### Option A: 1-Click Double Click (Recommended)
Simply double-click the included batch script:
```bat
run_unified_app.bat
```
This automatically configures the environment (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`, `CUDA_VISIBLE_DEVICES=-1`) and launches the native desktop GUI.

### Option B: Command Line Launch
```powershell
python unified_app.py
```

---

## 🖥️ Application Features & Tabs

### 1. 📊 Store Executive Dashboard (Tab 1)
* **Live Store KPIs:** Real-time Inflow, Outflow, Current Store Occupancy, Active Staff on Duty, and Patron-to-Staff service load ratio.
* **Dual Live Video Stream & Feeds:**
  * **Left:** Real-time Customer Footfall Camera Stream with virtual gates and a **RAM Saver Mode** toggle (`[ 👁️ Live Video: ON / OFF ]`).
  * **Right:** Real-time Staff Attendance stream with verification confidence, timestamp, and status.
* **Quick Action Controls:** 1-click Gate Calibrator launcher, 3-Photo Staff Enrollment modal, and Attendance CSV export.

### 2. 👥 Customer Analytics & Demographics (Tab 2)
* Live camera stream with virtual crossing gates (Entry, Exit, Dwell, Bag Check).
* RTSP camera configuration and sample video playback (`Outlet_Sample_1_Banasree.avi`).
* Real-time gate crossing audit table.

### 3. 👤 Staff Attendance & VisionGuard AI (Tab 3)
* Real-time video preview with YuNet facial landmark overlays and status badges.
* **Interactive Staff Roster:** Search by staff ID/Name, view last-seen timestamps, and view verification statuses.
* **3-Photo Multi-Angle Registration:** Captures multi-view portrait photos, auto-corrects EXIF rotation, scales images, and extracts a normalized centroid embedding for higher recognition accuracy.
* **Exportable Data Tables:** Instant CSV export for daily attendance records and quality inspection audit logs.

### 4. ⚙️ Central Settings & Diagnostics (Tab 4)
* Camera RTSP URL settings for both Customer Analytics and Staff Attendance.
* Outlet Shop ID and Central API Gateway endpoint settings.
* **1-Click Edge Diagnostic Self-Test:** Validates ONNX sessions, OpenCV codecs, database schemas, and camera network connectivity.

---

## 🌐 Central API Configuration & Data Transmission

The pipeline automatically sends attendance punch events asynchronously via HTTP POST:

* **Endpoint:** `https://ai.prangroup.com/api/v1/attendance`
* **Fallback Endpoint:** `http://ai.prangroup.com:8021/api/v1/attendance`
* **Authentication Header:** `X-API-Key: <your_outlet_secret_token>`

### Sample Punch Payload
```json
{
  "staff_id": "100234",
  "employee_name": "Rahim Ahmed",
  "cdate": "20261018",
  "ctime": "091530",
  "camera_id": "CAM_01_ENTRANCE",
  "terminal_id": "111111",
  "location_id": 3,
  "confidence": 0.8845,
  "status": "CONFIRMED",
  "punch_type": "AUTO"
}
```

> The transmission is performed on a background thread so the video pipeline never drops FPS even if the outlet internet connection experiences latency.

---

## 📐 Virtual Gate Calibration

To calibrate the virtual gates for your outlet camera feed:

1. In the Desktop GUI, open **Tab 2 (Customer Analytics)** and click **Launch Calibrator**, or run:
   ```powershell
   python FootFall_AgeGender/calibrate_gates.py
   ```
2. Click points on the video canvas to draw the **Entry**, **Exit**, and **Dwell** gate polygons.
3. Press `S` to save. Polygon coordinates are automatically updated in `FootFall_AgeGender/config/gates.json`.

---

## 📦 Converting to Standalone Windows Executable (.exe)

Because data is pushed over the REST API without direct Oracle HRIS dependencies, packaging into an `.exe` is completely self-contained.

Detailed instructions, PyInstaller flags, and the `.spec` template are located in:
👉 **[documentation/deployment_and_exe_guide.md](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/deployment_and_exe_guide.md)**

---

## 📚 Technical Documentation Hub

For in-depth architectural specifications, diagrams, and API definitions, explore the dedicated documentation folder:

* 📖 **[Documentation Index](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/README.md)**
* 🌲 **[Project Structure Breakdown](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/project_structure.md)**
* 📊 **[Architecture & Workflow Diagrams](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/architecture_and_diagrams.md)**
* 📡 **[Pipeline & REST API Specification](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/pipeline_and_api_spec.md)**
* 💻 **[Deployment & Executable Packaging Manual](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/deployment_and_exe_guide.md)**
