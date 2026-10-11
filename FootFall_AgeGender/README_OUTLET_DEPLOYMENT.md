# Multi-Outlet Deployment & Shop ID Configuration Guide

This guide details how to reconfigure the **Shop ID**, **API credentials**, and **camera stream** when deploying this project to a new supershop outlet or showroom.

---

## 📌 Overview

The Retail AI system communicates with a centralized **FastAPI Gateway** (`http://ai.prangroup.com:8021`). Every telemetry event generated at the edge—including customer entries, exits, bag detections, dwell times, and demographics—is tagged with a unique numerical **`shop_id`**.

When deploying this project to a new outlet, you must update the Shop ID so that all recorded analytics are attributed to the correct store in the central database.

```
┌────────────────────────────────────────┐
│     New Supershop Outlet (Edge PC)     │
│  - RetailAI Engine / app.py            │
│  - Configured with Outlet SHOP_ID      │
└───────────────────┬────────────────────┘
                    │ HTTP POST /api/v1/... (tagged with shop_id)
                    ▼
┌────────────────────────────────────────┐
│   Central FastAPI Gateway (Port 8021)   │
│   (Validates API Key & routes data)    │
└───────────────────┬────────────────────┘
                    ▼
┌────────────────────────────────────────┐
│    Central PostgreSQL Database         │
│    (Separates metrics by shop_id)      │
└────────────────────────────────────────┘
```

---

## 📋 Information Required Before Deployment

Before setting up the new outlet PC, request the following details from the central system administrator:

| Parameter | Example Value | Description |
| :--- | :--- | :--- |
| **Shop ID** | `4` | The integer ID registered in the central `shops` database table. |
| **API Key** | `secret_outlet_token_mirpur` | The authorized secret token for this specific branch. |
| **Camera RTSP URL** | `rtsp://admin:pass@192.168.1.100:554/...` | Local LAN RTSP stream URL from the outlet's NVR or IP camera. |
| **Camera ID** | `cam_01` | Identifier for the entrance camera at this branch. |

---

## 🛠️ Step-by-Step Configuration by Deployment Mode

### Mode 1: Standalone Windows Executable Deployment (`C:\RetailAI`)
*(Recommended for store Mini PCs with no Python installed)*

When deploying the standalone pre-compiled package (`dist/RetailAI` or `C:\RetailAI`):

#### 1. Edit `config/production.yaml`
Open `C:\RetailAI\config\production.yaml` in Notepad and modify lines 65–67:

```yaml
shop:
  shop_id: 4          # <--- SET NEW SHOP ID HERE
  camera_id: "cam_01" # Identifier for this branch camera

api:
  enabled: true
  base_url: "http://ai.prangroup.com:8021"
  api_key: "secret_outlet_token_mirpur"  # <--- SET NEW OUTLET API KEY HERE
  occupancy_interval_minutes: 20.0
  metrics_sync_interval_seconds: 5.0
```

#### 2. Update the Camera RTSP URL
In the same file (`C:\RetailAI\config\production.yaml`), update the camera stream URL under `camera:`:

```yaml
camera:
  rtsp_url: "rtsp://admin:pass@192.168.1.50:554/Streaming/Channels/101" # <--- NEW CAMERA RTSP
```

#### 3. Test & Recalibrate Virtual Gates
1. Double-click **`run_cmd.bat`** to run the interactive setup wizard.
2. Confirm the camera stream displays properly.
3. Type **`Y`** when prompted to calibrate gates for the new outlet entrance.
4. Save gates with the **`S`** key.

#### 4. Register Background Service
Right-click **`install_service.bat`** and select **"Run as administrator"**. The engine will now run 24/7 with the new Shop ID.

---

### Mode 2: Python Source Code Deployment (`app.py`)
*(For developers or PCs running directly from Python 3.12)*

If you run the project from source code:

#### 1. Edit `config/production.yaml`
Update `config/production.yaml` lines 65–67:
```yaml
shop:
  shop_id: 4
  camera_id: "cam_01"
api:
  api_key: "secret_outlet_token_mirpur"
```

#### 2. Edit the Root `.env` File
Open `.env` in the repository root:
```ini
API_GATEWAY_URL=http://ai.prangroup.com:8021
API_KEY=secret_outlet_token_mirpur
SHOP_ID=4
```

#### 3. If Using `run_service.bat` (Unattended Loop)
If you run the app using `run_service.bat`:
1. Copy `deploy/edge.env.example.bat` to `edge.env.bat` in the root folder.
2. Open `edge.env.bat` and set:
   ```bat
   set API_GATEWAY_URL=http://ai.prangroup.com:8021
   set API_KEY=secret_outlet_token_mirpur
   set SHOP_ID=4
   ```

---

### Mode 3: User Dashboard Deployment (`User-dashboard`)
*(Web/Desktop Reporting UI)*

If the new outlet PC also runs the local analytics dashboard (`User-dashboard`):

#### 1. Edit `User-dashboard/backend/.env` & `User-dashboard/.env`
Open `User-dashboard/backend/.env` and update:

```ini
# Central FastAPI Gateway Service
RETAIL_API_BASE_URL=http://ai.prangroup.com:8021
RETAIL_API_KEY=secret_outlet_token_mirpur

API_BASE_URL=http://ai.prangroup.com:8021
UPSTREAM_API_URL=http://ai.prangroup.com:8021
UPSTREAM_API_KEY=secret_outlet_token_mirpur

# Default Showroom/Shop ID for the dashboard
SHOP_ID=4

# Set LOCK_SHOP_ID=true to lock UI strictly to this outlet, or false to allow switching
LOCK_SHOP_ID=true
```

> [!NOTE]
> `User-dashboard/backend/config/db_config.py` automatically detects changes to `.env` files via file modification time and hot-reloads the active `SHOP_ID` without needing to restart the FastAPI backend.

---

## 📐 Mandatory: Calibrate Virtual Gates for the New Outlet

Every outlet has a different door width, camera angle, and focal length. **Never reuse `config/gates.json` from a previous shop.**

1. Run the gate calibration tool:
   - In Standalone mode: Run **`run_cmd.bat`** and choose **`Y`** when prompted for calibration.
   - In Python mode: Run `python calibrate_gates.py --source "rtsp://..."`
2. Calibrate the virtual lines:
   - **`1` (Footfall Gate)**: Draw line across the primary doorway threshold.
   - **`2` (Bag Exit Gate)**: Draw line where shopping bags are verified.
   - **`3` (Demographics Gate)**: Draw capture zone for face/body profiling.
   - **Inside Reference**: Click inside the shop to set the orientation arrow pointing **INTO** the store.
   - Press **`S`** to save coordinates to `config/gates.json`.
   - Press **`Q`** or **`Esc`** to exit.

---

## 🔍 Verification & Health Checklist

After changing the Shop ID and launching the application, perform these checks:

### 1. Check Edge Execution Logs
Open `retail_ai.log` and verify the initialization line:
```text
[ApiManager] Gateway: http://ai.prangroup.com:8021 | Outlet ID: 4 | Status: ONLINE
```
Ensure that `Outlet ID:` reflects your new `shop_id` and the status is `ONLINE`.

### 2. Verify Central API Gateway Connectivity
In a browser or via PowerShell:
```powershell
# Check health
curl http://ai.prangroup.com:8021/health

# Verify shop registration
curl http://ai.prangroup.com:8021/api/v1/shops
```
Ensure your new `shop_id` appears in the list returned by `/api/v1/shops`.

### 3. Verify Live Event Ingestion
Cross the virtual gate line on camera, then check `retail_ai.log`:
```text
>>> VALIDATED ENTRY RECORDED: Track 1 (Total Valid In=1)
[ApiManager] Dispatching entry event for shop_id=4, track_id=1
```
Check that the central gateway accepted the event without HTTP 401 (invalid key) or HTTP 400/422 (invalid shop ID) errors.

---

## 📁 File Reference Matrix

| Component | Target File | Key / Variable | Default in Repo |
| :--- | :--- | :--- | :--- |
| **Edge Engine (Config)** | [`config/production.yaml`](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/config/production.yaml#L65) | `shop.shop_id` | `3` |
| **Edge Engine (EXE)** | [`dist/RetailAI/config/production.yaml`](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/dist/RetailAI/config/production.yaml#L65) | `shop.shop_id` | `3` |
| **Edge Engine (Env)** | [`.env`](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/.env#L3) | `SHOP_ID` | `2` |
| **Windows Service Env** | [`deploy/edge.env.example.bat`](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/deploy/edge.env.example.bat#L5) | `set SHOP_ID=` | `2` |
| **Dashboard Backend Env**| [`User-dashboard/backend/.env`](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/User-dashboard/backend/.env#L20) | `SHOP_ID` | `3` |
| **Dashboard Root Env** | [`User-dashboard/.env`](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/User-dashboard/.env#L20) | `SHOP_ID` | `3` |
| **Virtual Gate Coordinates** | [`config/gates.json`](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/config/gates.json) | Door coordinates | Custom per store |
