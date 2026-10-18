# OmniRetail AI Suite — Technical Documentation Hub
=====================================================

Welcome to the central technical documentation repository for the **OmniRetail AI Suite** (Customer Traffic Analytics & VisionGuard Staff Attendance System).

This folder contains all design specifications, system diagrams, pipeline workflows, API contracts, database schemas, and deployment guides.

---

## 📑 Documentation Index

| Document | Description |
| :--- | :--- |
| **[1. Project Structure](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/project_structure.md)** | Comprehensive directory tree detailing all source code, models, configs, assets, and database files across both sub-engines. |
| **[2. Architecture & Workflow Diagrams](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/architecture_and_diagrams.md)** | Visual Mermaid diagrams including the overall System Architecture, End-to-End Data Pipeline, 8-Stage Face Recognition Workflow, and Virtual Gate Tracking Flow. |
| **[3. Pipeline & API Specifications](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/pipeline_and_api_spec.md)** | Technical reference for REST API contracts (`POST /api/v1/attendance`, `POST /api/v1/events/footfall`), network fallback logic, and SQLite database schemas. |
| **[4. Deployment & Executable (.exe) Guide](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/CustomerAnalytics_Staff_Attendence/documentation/deployment_and_exe_guide.md)** | Complete production guide for compiling the system into a standalone Windows `.exe` using PyInstaller, model asset bundling, and running as a 24/7 edge service. |

---

## 🎯 Target System Overview

* **Hardware Environment:** Intel Mini PC (4 Cores, 2.0+ GHz, 4–8 GB RAM, Integrated Intel HD Graphics, No Discrete GPU).
* **Operating System:** Windows 10 / Windows 11 / Windows Server 2022 (64-bit).
* **Network Topology:** Local Edge Camera RTSP over TCP LAN $\rightarrow$ Local AI Processing $\rightarrow$ Outbound HTTPS to Central Cloud API Gateway (`ai.prangroup.com`).
* **Storage Footprint:** ~500 MB for Python code, ONNX models, and static assets; SQLite database and face crops occupy ~1–2 GB after months of operation.
