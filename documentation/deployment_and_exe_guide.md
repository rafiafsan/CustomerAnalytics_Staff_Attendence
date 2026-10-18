# Deployment & Executable (.exe) Packaging Manual
===================================================

This manual explains how to compile the **OmniRetail AI Suite** into a standalone Windows executable (`.exe`) using **PyInstaller**, deploy it to retail edge Mini PCs, and configure it for unattended 24/7 operation.

---

## 1. Why PyInstaller Packaging is Now Simplified

Because attendance data is routed through the central REST API (`https://ai.prangroup.com/api/v1/attendance`), there is **no requirement for Oracle Instant Client binaries, Oracle DLLs, or `oracledb` C-extensions** on the edge PC. 

The application is composed entirely of pure Python libraries and cross-platform C-runtimes (OpenCV, ONNX Runtime, PyTorch CPU).

---

## 2. Directory Mode (`--onedir`) vs Single File (`--onefile`)

> [!IMPORTANT]
> **Always use `--onedir` (Folder distribution) for Computer Vision and AI applications.**

| Mode | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **`--onedir` (Folder)** | • **Instant startup** (0–1s)<br/>• Fast model weight loading<br/>• Easy to update config files and ONNX models without recompiling | Produces a folder containing the `.exe` alongside dependent `.dll` and model files | **RECOMMENDED (Production)** |
| **`--onefile` (Single .exe)** | Single `.exe` binary file | • **15–30 second launch delay** on every startup (must unpack 500 MB into `%TEMP%`)<br/>• High disk thrashing on low-cost Mini PCs | Not recommended for production |

---

## 3. Step-by-Step Compilation Guide

### Step 1: Install PyInstaller in your Virtual Environment
Ensure your virtual environment is active:
```powershell
pip install pyinstaller
```

### Step 2: Prepare `OmniRetail.spec`
Create a PyInstaller specification file in the root workspace directory named `OmniRetail.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
PROJECT_ROOT = Path('.').resolve()

# Assets and models to bundle
datas = [
    # 1. Attendence System models and assets
    ('Attendence System/models', 'Attendence System/models'),
    ('Attendence System/data', 'Attendence System/data'),
    ('Attendence System/static', 'Attendence System/static'),
    ('Attendence System/templates', 'Attendence System/templates'),
    ('Attendence System/config.py', 'Attendence System'),
    ('Attendence System/database.py', 'Attendence System'),

    # 2. FootFall_AgeGender models and configs
    ('FootFall_AgeGender/models', 'FootFall_AgeGender/models'),
    ('FootFall_AgeGender/config', 'FootFall_AgeGender/config'),
    ('FootFall_AgeGender/database', 'FootFall_AgeGender/database'),
    ('FootFall_AgeGender/mivolo_system', 'FootFall_AgeGender/mivolo_system'),
]

hiddenimports = [
    'PIL._tkinter_finder',
    'cv2',
    'numpy',
    'requests',
    'yaml',
    'supervision',
    'onnxruntime',
    'torch',
    'torchvision',
    'ultralytics',
    'timm',
    'psutil',
    'sqlite3',
]

a = Analysis(
    ['unified_app.py'],
    pathex=[
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / 'FootFall_AgeGender'),
        str(PROJECT_ROOT / 'Attendence System'),
        str(PROJECT_ROOT / 'FootFall_AgeGender' / 'mivolo_system')
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['oracledb', 'test', 'tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OmniRetail_Suite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for windowed production release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OmniRetail_Suite',
)
```

### Step 3: Run the PyInstaller Build
Execute the build command from PowerShell:
```powershell
pyinstaller OmniRetail.spec --noconfirm --clean
```

Upon completion, your self-contained distribution folder will be generated at:
```
dist/OmniRetail_Suite/
│
├── OmniRetail_Suite.exe     # The compiled executable
├── Attendence System/       # Bundled models & configs
├── FootFall_AgeGender/      # Bundled YOLO & gate configs
└── python3*.dll & torch C-runtimes
```

---

## 4. Hardware Sizing & Recommended Edge Specifications

| Specification | Minimum Requirement | Recommended Production Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i3 (4 Cores, 2.0 GHz) | Intel Core i5 / Intel N100 (4 Cores, 3.4 GHz turbo) |
| **System Memory (RAM)** | 4 GB DDR4 | 8 GB DDR4 / DDR5 |
| **Graphics (GPU)** | Not Required (CPU Inference) | Intel UHD Graphics (OpenVINO accelerated) |
| **Storage (Disk)** | 64 GB SSD (SATA) | 128 GB NVMe SSD |
| **Network Interface** | 100 Mbps Ethernet LAN | 1 Gbps Ethernet LAN |
| **Operating System** | Windows 10 Pro (64-bit) | Windows 10/11 Pro (64-bit) |

---

## 5. Unattended 24/7 Autostart via Windows Task Scheduler

To ensure the application starts automatically when the outlet PC boots up and auto-recovers from unexpected power cuts:

1. Open **Windows Task Scheduler** (`taskschd.msc`).
2. Click **Create Task**:
   * **General:** Name: `OmniRetail_AI_Suite`, check *"Run whether user is logged on or not"*, check *"Run with highest privileges"*.
   * **Triggers:** New $\rightarrow$ Begin the task: *"At startup"*, Delay task for: `30 seconds` (to ensure camera network LAN is initialized).
   * **Actions:** New $\rightarrow$ Action: *"Start a program"*, Program/script: `C:\RetailAI\dist\OmniRetail_Suite\OmniRetail_Suite.exe` (or your batch file `run_unified_app.bat`), Start in: `C:\RetailAI\dist\OmniRetail_Suite\`.
   * **Conditions:** Uncheck *"Start the task only if the computer is on AC power"*.
   * **Settings:** Check *"If the task fails, restart every 1 minute"*, attempt restart up to `99 times`.
