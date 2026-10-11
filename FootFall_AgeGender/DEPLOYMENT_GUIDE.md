# Retail AI Analytics Engine — Edge Deployment Guide

A fully standalone, zero-dependency Footfall, Demographics & Analytics system packaged for Windows.  
**No Python, pip packages, compilers, or external runtimes are required on the target machine.**

---

## 📋 System Requirements

* **Operating System**: Windows 10, Windows 11, or Windows Server (64-bit)
* **Hardware**: Intel or AMD 64-bit CPU (4+ cores recommended, ~2 GB free RAM)
* **GPU**: None required (runs on CPU)
* **Network**:
  * Local LAN access to NVR / IP Camera RTSP stream (Port 554)
  * Outbound HTTP/HTTPS access to Central API Gateway (`http://ai.prangroup.com:8021`)
* **Permissions**: Local Administrator rights (only needed once to register the background service)

---

## 📂 Package Contents

When you copy the `RetailAI` folder to a target PC, it includes:

| File / Folder | Description |
| :--- | :--- |
| **`RetailAI.exe`** | The core standalone engine containing bundled Python, ONNX Runtime, PyTorch CPU, OpenCV codecs, and tracking models. |
| **`run_cmd.bat`** | **Interactive Setup Wizard**: Run this in CMD for initial camera configuration, gate calibration, and live video preview. |
| **`install_service.bat`** | **1-Click Service Installer**: Registers RetailAI as an automatic Windows Startup service running 24/7 in the background. |
| **`uninstall_service.bat`** | **Service Uninstaller**: Stops and cleanly removes the Windows Startup service. |
| **`check_service_status.bat`**| Checks whether the background service is active, running, or stopped. |
| **`run_service.bat`** | Helper script used by the Windows Service to run headless without a GUI window. |
| **`config/`** | Configuration folder containing `production.yaml` (outlet ID, RTSP URL, API token) and `gates.json` (virtual gate coordinates). |
| **`models/`** | Bundled YOLO ONNX person and bag detection weights. |
| **`mivolo_system/`** | Bundled MiVOLO age & gender demographic profiling model. |
| **`retail_ai.log`** | Live execution log file capturing health metrics, footfall counts, and API transmission statuses. |

---

## 🚀 Step-by-Step Installation & Setup

### **Step 1: Copy Folder to Target PC**

1. Copy the entire `dist/RetailAI` folder to a dedicated directory on the target PC.
2. **Recommended Path**: `C:\RetailAI`  
   *(Avoid placing in folders with very long names or restricted permissions like Desktop of a temporary user).*

---

### **Step 2: Configure Store ID & API Credentials for the Outlet**

Every physical supershop has a designated **Shop ID** and unique **API Token** registered on the Central FastAPI Gateway (`http://ai.prangroup.com:8021`).

#### **A. For Standalone Executable Deployment (`C:\RetailAI`):**
Open `C:\RetailAI\config\production.yaml` in Notepad:

```yaml
shop:
  shop_id: 3          # Set to the target outlet's unique Shop ID (e.g. 1, 2, 3, 4...)
  camera_id: "cam_01" # Identifier for this entrance camera (e.g. cam_01, entrance_cam)

api:
  enabled: true
  base_url: "http://ai.prangroup.com:8021" # Central company gateway
  api_key: "secret_outlet_token_banasree"  # Replace with the authorized API Token for this outlet
```
Save and close the file.

> [!IMPORTANT]
> The Shop ID and API Key must match the outlet registration in the central system. The Edge AI engine embeds this `shop_id` in every validated customer Entry, Exit, Dwell time, and Demographic event.

#### **B. For Source Code Deployment (`python app.py`):**
If deploying directly from the Python repository source:
1. Update `config/production.yaml` (lines 65–68) as shown above.
2. Update the root `.env` file:
   ```ini
   API_GATEWAY_URL=http://ai.prangroup.com:8021
   API_KEY=your_outlet_secret_token
   SHOP_ID=3
   ```
3. If running unattended with `run_service.bat`, copy `deploy/edge.env.example.bat` to `edge.env.bat` and set `set SHOP_ID=3` and `set API_KEY=your_token`.

#### **C. For User Dashboard (Store or Head-Office Analytics):**
If this PC also hosts the web/desktop reporting dashboard (`User-dashboard`):
1. Open `User-dashboard/backend/.env` (and `User-dashboard/.env`):
   ```ini
   SHOP_ID=3
   LOCK_SHOP_ID=true   # Set to true to strictly lock this PC's dashboard to this outlet
   ```
2. The dashboard backend automatically detects and hot-reloads the updated `SHOP_ID` without restarting.

*(For full step-by-step guidance on all configuration files, see [README_OUTLET_DEPLOYMENT.md](file:///c:/Users/SOOQ%20ELASER/Desktop/New%20folder/PAL_DailyShopping/README_OUTLET_DEPLOYMENT.md)).*

---

### **Step 3: Initial Setup & Gate Calibration (Interactive Mode)**

1. Double-click **`run_cmd.bat`** (or open Command Prompt, navigate to `C:\RetailAI`, and run `RetailAI.exe`).
2. Follow the on-screen prompts:

#### **A. Camera RTSP Stream URL**
```text
[Current Source]: rtsp://ctrl:ctl%40123456@10.1.56.11:554/Streaming/Channels/1201
Enter NVR RTSP Stream URL or Video Path
(Press ENTER to keep current):
```
* Press **ENTER** to keep the existing camera link.
* **Or** type your new RTSP URL (e.g. `rtsp://admin:password@192.168.1.100:554/h264Preview_01_main`) and press Enter.  
  *(The new URL is automatically saved to `config/production.yaml`).*

#### **B. Virtual Gate Calibration**
```text
Do you want to calibrate virtual gates for this camera? [Y/n]:
```
* Type **`Y`** and press Enter.
* An interactive calibration window will open displaying the live camera feed:
  * **Select Gate Type**:
    * Press **`1`**: Footfall Gate (Yellow) — entrance/exit counting line.
    * **`2`**: Bag Exit Gate (Cyan) — bag verification line.
    * **`3`**: Demographics Gate (Green) — age/gender capture zone.
    * **`0`**: Master Gate (White).
  * **Draw Line**: Click and drag your mouse on the video feed to draw the gate line across the doorway.
  * **Inside Direction**: Click on the side of the doorway that corresponds to the **INSIDE** of the shop (an arrow will point inside).
  * **Hotkeys**:
    * **`S`**: Save configuration.
    * **`R`**: Reset currently selected gate.
    * **`C`**: Clear all gates.
    * **`Q`** or **`Esc`**: Save and exit calibration.

#### **C. Live Video Preview**
```text
Show live video preview window? [Y/n]:
```
* Press **ENTER** (or `Y`) to launch the live video window.
* Verify that customers crossing the yellow line are counted as **`ENTRY`** or **`EXIT`**.
* Press **`Q`** on the video window when you are satisfied with tracking.

---

### **Step 4: Install as 24/7 Background Windows Service**

Once the camera and virtual gates are tested:

1. Right-click **`install_service.bat`**.
2. Select **"Run as administrator"**.
3. A command prompt will confirm:
   ```text
   [SUCCESS] RetailAI Service registered successfully!
   Starting the background service now...
   Service is now running in the background!
   ```
4. **What this does**:
   * Registers a Windows Scheduled Task (`RetailAI_Service`) under `NT AUTHORITY\SYSTEM`.
   * Configured to **start automatically at system startup** before any user logs in.
   * Runs headless in the background (no black CMD window remains on screen).
   * **Auto-recovery**: If the PC restarts, experiences power failure, or wakes from sleep, the service resumes automatically.
   * If the process ever exits unexpectedly, Windows restarts it within 60 seconds (`RestartCount 999`).

---

## 🔍 Verification & Health Monitoring

### **1. Check Background Service Status**
Double-click **`check_service_status.bat`**:
* **State**: `Running` indicates the engine is actively processing camera frames.
* **Last Result**: `0` indicates healthy execution.

### **2. Check Live Execution Logs**
Open `retail_ai.log` in Notepad:
* Look for:
  ```text
  [ApiManager] Gateway: http://ai.prangroup.com:8021 | Outlet ID: 3 | Status: ONLINE
  [UnifiedPipeline] Running in Headless Mode (Zero GUI Overhead). Streaming events to API Gateway...
  [Health] uptime=30s ai_fps=1.4 inf_ms=210 ram=850MB cpu=260% cam=ON reconnects=0
  >>> VALIDATED ENTRY RECORDED: Track 12 (Total Valid In=5)
  >>> VALIDATED EXIT RECORDED: Track 14 (Total Valid Out=4)
  ```

### **3. Check Central API Gateway**
Visit `http://ai.prangroup.com:8021/health` in your browser to verify server connectivity.

---

## 🛠️ Management & Maintenance

### **How to Recalibrate Gates or Change Camera URL Later**
1. Stop the background service by running **`uninstall_service.bat`** (as administrator).
2. Run **`run_cmd.bat`** to enter the new RTSP link or redraw gate lines.
3. Once finished, run **`install_service.bat`** (as administrator) to reactivate the background service.

### **How to Uninstall or Remove the Service**
1. Right-click **`uninstall_service.bat`** and select **"Run as administrator"**.
2. The service is stopped and unregistered immediately.

---

## ❓ Troubleshooting

| Issue | Cause & Solution |
| :--- | :--- |
| **`Could not open video source`** | The RTSP stream URL is incorrect or unreachable. Verify camera IP, username, password, and port 554 in VLC media player first. |
| **Service does not start with `install_service.bat`** | You must right-click and select **"Run as administrator"**. Standard user permissions cannot register system startup tasks. |
| **`retail_ai.log` reports `HTTP 401 Unauthorized`** | The `api_key` in `config/production.yaml` does not match the token registered on the API Gateway. Update the key in `production.yaml`. |
| **High CPU usage on low-end Mini PC** | The system is auto-scheduled for 4 FPS AI inference on CPU. If CPU is consistently at 100%, reduce camera resolution in your NVR settings to 1280x720 or 640x360. |
| **Power loss recovery** | In the Mini PC BIOS settings, ensure **"Restore on AC Power Loss"** is set to **"Power On"** so the PC boots up immediately after power returns. |
