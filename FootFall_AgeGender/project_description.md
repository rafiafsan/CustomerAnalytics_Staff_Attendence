# Project Description — PAL DailyShopping Retail AI Analytics Engine

## Overview

**PAL DailyShopping Retail AI Analytics Engine** is a real-time, edge-deployed computer vision system purpose-built for retail supershop outlets. It runs entirely on low-resource hardware (Intel Mini PC, 4 CPU cores, ~2 GB RAM, **no GPU required**) and provides automated customer footfall counting, demographic profiling, and live store occupancy tracking — all without exposing any database directly to the network edge.

---

## Purpose

Retail chains operating multiple physical outlets need accurate, automated insights into customer behaviour without deploying expensive GPU servers at every branch. This system solves that by:

- Running a full AI inference pipeline locally on a cheap Windows Mini PC at each outlet.
- Transmitting validated analytics events to a **central cloud/on-premise gateway** over HTTP — keeping the database safely behind the company firewall.
- Providing store managers and head-office with live dashboards, demographic breakdowns, and daily footfall reports.

---

## Key Capabilities

| Feature | Detail |
| :--- | :--- |
| **Bidirectional Footfall Counting** | Counts customer entries and exits across a configurable virtual gate drawn over the camera feed. |
| **Anti-Bounce Dwell Validation** | Entry events require 60-second dwell; exit events require 10-second dwell — eliminating false triggers from loitering staff. |
| **Demographic Profiling** | MiVOLO deep-learning model classifies age group and gender per customer (lazy-loaded, CPU-only). |
| **Live Occupancy Snapshots** | Current in-store visitor count is pushed to the central database every 20 minutes. |
| **Multi-Outlet Support** | Each outlet is assigned a unique `shop_id` and `api_key`; all events are tagged and routed accordingly. |
| **Zero-Drop Event Queue** | A non-blocking in-memory queue decouples real-time video processing from HTTP transmission — no frame is dropped due to network latency. |
| **24/7 Background Windows Service** | Installs as a Windows Scheduled Task that auto-starts on boot and auto-recovers from crashes within 60 seconds. |
| **Standalone Executable** | Distributed as a self-contained `.exe` — no Python, pip, or compiler needed on the outlet PC. |

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Detection Model** | YOLOv8m ONNX (320x320, CPU-optimised) via ONNX Runtime |
| **Person Tracking** | ByteTrack multi-object tracker |
| **Demographics** | MiVOLO age & gender classification |
| **Video Input** | OpenCV RTSP / local video file (TCP transport) |
| **API Transport** | Async HTTP POST to FastAPI Gateway (non-blocking queue) |
| **Backend Gateway** | FastAPI + Uvicorn (central server, Port 8021) |
| **Database** | PostgreSQL (central; never directly accessed from edge) |
| **Configuration** | YAML (`production.yaml`) + JSON (`gates.json`) + `.env` |
| **Packaging** | PyInstaller standalone bundle |
| **Language** | Python 3.12 |

---

## System Architecture

```
     Retail Outlet (Edge Mini PC)                      Company Central Server
  +-------------------------------+               +-------------------------------+
  |         CCTV Camera           |               |                               |
  |     (RTSP / Video File)       |               |                               |
  +---------------+---------------+               |                               |
                  |                               |                               |
                  v                               |                               |
  +-------------------------------+               |                               |
  |     RetailAI.exe / app.py     |               |                               |
  |  - YOLOv8 ONNX (320x320)      |   HTTP POST   |  FastAPI Gateway (Port 8021)  |
  |  - ByteTrack Person Tracking  +-------------->|  http://ai.prangroup.com:8021 |
  |  - MiVOLO Demographics        |  (Non-blocking|                               |
  |  - Non-blocking Event Queue   |   Queue)      +---------------+---------------+
  +-------------------------------+                               |
                                                                  v
                                                  +-------------------------------+
                                                  |      PostgreSQL Database       |
                                                  |  * customer_entries            |
                                                  |  * customer_exits              |
                                                  |  * customer_demographics       |
                                                  |  * store_occupancy             |
                                                  |  * daily_metrics               |
                                                  +-------------------------------+
```

---

## Target Hardware

- **OS**: Windows 10 / 11 (64-bit)
- **CPU**: Intel Celeron / Pentium / Core i3 or i5 (4 cores)
- **RAM**: ~4 GB physical (~2 GB usable for this process)
- **GPU**: None required
- **Network**: LAN access to IP camera (RTSP port 554) + outbound HTTP to gateway

---

## Deployment Model

Each retail outlet runs an instance of this engine configured with:
- A unique **Shop ID** (`shop_id` in `production.yaml`)
- A unique **API Token** (`api_key`) for authenticated event submission
- Custom **virtual gate coordinates** (`gates.json`) drawn over the outlet's specific camera angle

The engine can be deployed in three modes:
1. **Standalone EXE** — copy `RetailAI/` folder, run `run_cmd.bat` for setup, then `install_service.bat` as Administrator.
2. **Python Source** — `py -3.12 app.py [--source <rtsp_url | video_file>]`
3. **Headless Background Service** — auto-starts on Windows boot; recovers automatically after power cuts or crashes.

---

## Analytics Events Recorded

| Event Type | Trigger |
| :--- | :--- |
| `customer_entry` | Person crosses gate inward **and** dwells >= 60 s inside |
| `customer_exit` | Person crosses gate outward **and** dwells >= 10 s outside |
| `customer_demographics` | Age group + gender classification per validated customer |
| `store_occupancy` | Periodic snapshot of live visitor count (every 20 min) |
| `daily_metrics` | Aggregated daily footfall summary |

---

## Project Structure (Key Modules)

```
PAL_DailyShopping/
+-- app.py                      # Main application entry point
+-- config/
|   +-- production.yaml         # Outlet ID, RTSP URL, API credentials, thresholds
|   +-- gates.json              # Virtual gate coordinates per camera
|   +-- database.json           # Central database credentials (server-side only)
+-- camera/rtsp_reader.py       # Low-latency RTSP / video capture thread
+-- inference/onnx_detector.py  # ONNX Runtime CPU multi-threaded YOLO detector
+-- tracking/track_manager.py   # ByteTrack multi-object tracker
+-- gate/event_manager.py       # Dwell validation & entry/exit event logic
+-- mivolo_system/async_worker.py # Asynchronous MiVOLO demographics profiler
+-- database/db_manager.py      # Non-blocking HTTP API Gateway event sender
+-- services/unified_pipeline.py# Central analytics pipeline coordinator
+-- User-dashboard/             # Web/desktop analytics reporting dashboard
+-- deploy/                     # Service installer scripts (BAT files)
+-- dist/RetailAI/              # Pre-compiled standalone distribution package
+-- DEPLOYMENT_GUIDE.md         # Step-by-step outlet deployment instructions
+-- README_OUTLET_DEPLOYMENT.md # Multi-outlet configuration reference
+-- README.md                   # Full technical documentation
```

---

## Related Documentation

| Document | Purpose |
| :--- | :--- |
| [README.md](README.md) | Full technical reference, configuration table, build instructions |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Step-by-step guide for deploying the EXE to a new outlet PC |
| [README_OUTLET_DEPLOYMENT.md](README_OUTLET_DEPLOYMENT.md) | Multi-outlet Shop ID, API key, and camera reconfiguration |
| [run.md](run.md) | Quick-start commands for running the system |
