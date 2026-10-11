# OmniRetail AI Suite — Unified Enterprise Desktop Gateway

An integrated, high-performance **Windows Desktop GUI Application** uniting edge customer traffic analytics and staff facial recognition attendance into a single, cohesive management interface.

---

## 🎯 Architecture Overview

This unified system merges two standalone edge vision pipelines with **zero web server dependencies** and **zero dropped features**:

```
                               ┌────────────────────────────────────────────────────────┐
                               │           OmniRetail AI Suite (unified_app.py)          │
                               │                Native Dark Windows GUI                 │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
          ┌───────────────────────────────────────────────┴──────────────────────────────────────────────┐
          │                                                                                              │
┌─────────▼─────────────────────────────────────┐                              ┌─────────────────────────▼─────────────────────────────────────┐
│ 👥 Customer Analytics & Demographics          │                              │ 👤 VisionGuard Staff Attendance                               │
│ (FootFall_AgeGender)                          │                              │ (Attendence System)                                           │
├───────────────────────────────────────────────┤                              ├───────────────────────────────────────────────────────────────┤
│ • Model: YOLOv8 ONNX (320x320) + ByteTrack    │                              │ • Model: YuNet ONNX (Face & 5 Landmarks)                      │
│ • Virtual Gates: Entry, Exit, Dwell, Bags     │                              │ • Alignment & Quality: Sharpness, Yaw, Roll, Brightness       │
│ • Profiling: MiVOLO Age & Gender Estimation   │                              │ • Embeddings: SFace 128-d L2 Normalized Vectors               │
│ • Local Edge DB: SQLite `edge_events.db`      │                              │ • Verification: Cosine Similarity >= 0.380 + Multi-frame      │
│ • Cloud API Sync: Public REST Gateway         │                              │ • Local Edge DB: SQLite `attendance.db` + Photo Gallery       │
│ • Interactive Virtual Gate Calibration Tool   │                              │ • Interactive 3-Photo Registration Modal with EXIF Fix        │
└───────────────────────────────────────────────┘                              └───────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run the Unified Desktop Application

### Option 1: Double-Click Batch File (Recommended)
Simply double-click:
```bat
run_unified_app.bat
```

### Option 2: Command Line (PowerShell / Command Prompt)
From the root workspace directory:
```bash
python unified_app.py
```

---

## 🖥️ Application Features & Layout

### 1. 📊 Store Executive Dashboard (Tab 1)
- **Consolidated Store KPIs**:
  - `Customer Inflow`: Total validated store entries
  - `Customer Outflow`: Total validated store exits
  - `Current Occupancy`: Real-time patrons inside the outlet
  - `Staff on Duty`: Checked-in staff vs Total enrolled staff roster
  - `Patron : Staff Ratio`: Live service load ratio (patrons per active staff member)
  - `Bag Exit Crossings`: Shoppers exiting with validated bags
- **Quick Action Bar**:
  - Start/Stop Customer Footfall Monitoring
  - Launch Virtual Gate Calibrator
  - Enroll New Staff Member (3-Photo registration)
  - Run SFace Face Verifier Tester
  - Export Daily Attendance CSV
- **Dual Live Activity Feeds & Live Camera Video Stream**:
  - Left: **Live Customer Footfall Camera Stream & Virtual Gates Canvas** with user toggle `[ 👁️ Live Video: ON / OFF (RAM Saver) ]`. Displays the real-time camera feed with virtual gates, bounding boxes, and dwell HUD. When toggled OFF, canvas rendering halts completely, reducing RAM and CPU utilization while background AI tracking continues uninterrupted. Below the video, the real-time gate crossings and dwell log displays live events.
  - Right: Real-time Staff Attendance stream with verification confidence, timestamp, and status.

### 2. 👥 Customer Analytics & Demographics (Tab 2)
- Direct embedded view of the entire `RetailAnalyticsApp`:
- Camera / NVR RTSP stream configuration with automatic YAML saving.
- Video file browser for recorded sample analysis (`Outlet_Sample_1_Banasree.avi`).
- Live Customer Footfall video canvas with virtual gates and RAM saver display toggle (`[ 👁️ Live Video: ON / OFF ]`).
- Interactive OpenCV Virtual Gate Calibrator launcher (`calibrate_gates.py`).
- 4 KPI cards: Total In, Total Out, Current Inside, Bag Exits.
- Real-time event log with zero dropped FPS.

### 3. 👤 Staff Attendance & VisionGuard AI (Tab 3)
- Direct embedded view of the entire `AttendanceApp`:
- Live camera video feed with OpenCV bounding boxes, YuNet landmark points, identity labels, and status badges.
- Stream sampling controls (Pause / Resume, Target FPS, RTSP reconnect).
- KPI cards: Frame Sampling FPS, Live Captures, Quality Pass Rate %, Enrolled Staff, Today Verified.
- Live detected face chips carousel with match score bars.
- Interactive Staff Roster:
  - Search filter by Name / ID.
  - Filter pills: `All`, `Verified Today`, `Pending`.
  - Enrolled avatar photo preview and last-seen indicator.
- 3-Photo Multi-Angle Staff Enrollment Modal (with thread-safe SFace embeddings, auto-rotation, downscaling).
- SFace Embedding Verification Tester modal.
- CSV-Exportable Data Tables:
  - Today's Attendance Records Table.
  - Quality Inspection Logs Table.
  - Live System Log Scroller.

### 4. ⚙️ Central Settings & Diagnostics (Tab 4)
- Centralized configuration for both cameras (Retail Footfall RTSP and Attendance RTSP).
- Cloud API Base URL and Outlet Shop ID.
- AI Recognition and Blur thresholds.
- **Edge Self-Test Diagnostic Tool**: Runs a comprehensive 1-click health check on Python packages, OpenCV, ONNX Runtime sessions, model weights, database schemas, and camera connectivity.

---

## 🔒 Backward Compatibility Guarantee

Both original sub-projects remain 100% functional independently:
- Running `FootFall_AgeGender\main_gui.py` continues to run the standalone Retail Analytics application.
- Running `Attendence System\main_gui.py` continues to run the standalone VisionGuard Attendance application.
- No files were deleted, and all internal logic remains untouched.
