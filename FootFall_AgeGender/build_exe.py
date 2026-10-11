import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist" / "RetailAI"

print("=" * 70)
print("  RETAIL AI STANDALONE EXECUTABLE BUILDER (PyInstaller onedir)")
print(f"  Output Directory: {DIST_DIR}")
print("=" * 70)

# 1. Run PyInstaller
cmd = [sys.executable, "-m", "PyInstaller", "--clean", "-y", "RetailAI.spec"]
print(f"Running: {' '.join(cmd)}")
ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
if ret.returncode != 0:
    print(f"\n[ERROR] PyInstaller failed with return code {ret.returncode}")
    sys.exit(ret.returncode)

print("\nPyInstaller compilation succeeded! Packaging runtime asset directories...")

# 2. Ensure runtime assets exist in dist/RetailAI
assets_to_copy = [
    ("config", DIST_DIR / "config"),
    ("models", DIST_DIR / "models"),
    ("mivolo_system/model", DIST_DIR / "mivolo_system" / "model"),
    ("test_videos", DIST_DIR / "test_videos"),
]

for src_rel, dst in assets_to_copy:
    src = PROJECT_ROOT / src_rel
    if src.exists():
        print(f"Copying {src_rel} -> {dst.relative_to(PROJECT_ROOT)}...")
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    else:
        print(f"Warning: Source asset directory {src} does not exist!")

# 2b. Ensure torchvision native extensions are present in _internal/torchvision
try:
    import torchvision
    tv_src = Path(torchvision.__file__).parent
    tv_dst = DIST_DIR / "_internal" / "torchvision"
    tv_dst.mkdir(parents=True, exist_ok=True)
    for ext in ("*.pyd", "*.dll"):
        for f in tv_src.glob(ext):
            shutil.copy2(f, tv_dst / f.name)
    print("Copied torchvision native C extensions into _internal/torchvision")
except Exception as exc:
    print(f"Warning: Failed to copy torchvision C extensions: {exc}")


# 3. Create interactive runner script (run_cmd.bat)
cmd_bat = """@echo off
title Retail AI Analytics - Interactive Setup & Live View
cd /d "%~dp0"
echo ============================================================
echo   Starting RetailAI in Interactive Setup Mode
echo ============================================================
RetailAI.exe
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo System exited with code %ERRORLEVEL%
    pause
)
"""
(DIST_DIR / "run_cmd.bat").write_text(cmd_bat, encoding="utf-8")
print("Created run_cmd.bat")

# 4. Create headless service runner script (run_service.bat)
service_bat = """@echo off
cd /d "%~dp0"
RetailAI.exe --service
"""
(DIST_DIR / "run_service.bat").write_text(service_bat, encoding="utf-8")
print("Created run_service.bat")

# 5. Create Windows Service Installer (install_service.bat)
install_bat = """@echo off
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ============================================================
    echo   ERROR: Administrator privileges required!
    echo   Please right-click install_service.bat and "Run as administrator".
    echo ============================================================
    pause
    exit /b 1
)

set "APP_DIR=%~dp0"
if "%APP_DIR:~-1%"=="\\" set "APP_DIR=%APP_DIR:~0,-1%"
set "EXE_PATH=%APP_DIR%\\RetailAI.exe"

echo ============================================================
echo   Installing RetailAI as Persistent Windows Startup Service
echo   App Directory: %APP_DIR%
echo   Executable:    %EXE_PATH%
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$action = New-ScheduledTaskAction -Execute '%EXE_PATH%' -Argument '--service' -WorkingDirectory '%APP_DIR%';" ^
  "$trigger = New-ScheduledTaskTrigger -AtStartup;" ^
  "$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable;" ^
  "Register-ScheduledTask -TaskName 'RetailAI_Service' -Action $action -Trigger $trigger -Settings $settings -User 'SYSTEM' -RunLevel Highest -Force;"

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] RetailAI Service registered successfully!
    echo Starting the background service now...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-ScheduledTask -TaskName 'RetailAI_Service'"
    echo.
    echo Service is now running in the background!
    echo It will automatically start every time Windows boots up or restarts.
) else (
    echo.
    echo [ERROR] Failed to register service. Error code: %errorlevel%
)
echo.
pause
"""
(DIST_DIR / "install_service.bat").write_text(install_bat, encoding="utf-8")
print("Created install_service.bat")

# 6. Create Windows Service Uninstaller (uninstall_service.bat)
uninstall_bat = """@echo off
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ============================================================
    echo   ERROR: Administrator privileges required!
    echo   Please right-click uninstall_service.bat and "Run as administrator".
    echo ============================================================
    pause
    exit /b 1
)

echo ============================================================
echo   Stopping and Removing RetailAI Windows Service...
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Stop-ScheduledTask -TaskName 'RetailAI_Service' -ErrorAction SilentlyContinue;" ^
  "Unregister-ScheduledTask -TaskName 'RetailAI_Service' -Confirm:$false -ErrorAction SilentlyContinue;"

echo.
echo [SUCCESS] RetailAI Service stopped and unregistered.
echo.
pause
"""
(DIST_DIR / "uninstall_service.bat").write_text(uninstall_bat, encoding="utf-8")
print("Created uninstall_service.bat")

# 7. Create Service Status Inspector (check_service_status.bat)
status_bat = """@echo off
echo ============================================================
echo   RetailAI Windows Service Status
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$t = Get-ScheduledTask -TaskName 'RetailAI_Service' -ErrorAction SilentlyContinue; if ($t) { $info = Get-ScheduledTaskInfo -TaskName 'RetailAI_Service'; Write-Host 'State:        ' $t.State; Write-Host 'Last Run:     ' $info.LastRunTime; Write-Host 'Last Result:  ' $info.LastTaskResult } else { Write-Host 'RetailAI_Service is NOT currently installed.' }"
echo.
pause
"""
(DIST_DIR / "check_service_status.bat").write_text(status_bat, encoding="utf-8")
print("Created check_service_status.bat")

# 8. Create Operator Quick Start README
readme_text = """RETAIL AI ANALYTICS - STANDALONE DEPLOYMENT GUIDE
=================================================
This package runs on clean Windows systems with NO Python, packages, or dependencies.

HOW TO USE:

1. FIRST-TIME SETUP & CALIBRATION:
   - Double-click or run 'run_cmd.bat' (or run 'RetailAI.exe' directly in CMD).
   - Enter your camera RTSP stream URL when prompted (it will be saved to config/production.yaml).
   - Choose 'Y' to launch the Virtual Gate Calibration tool to set your store lines.
   - Verify the live video preview window.

2. RUN AS A PERMANENT WINDOWS SERVICE:
   - Right-click 'install_service.bat' and select 'Run as administrator'.
   - The system is now registered to start automatically in the background whenever Windows boots or restarts.
   - No user needs to be logged in, and no window will appear.
   - Logs are continuously written to 'retail_ai.log'.

3. CHECK STATUS OR STOP:
   - Run 'check_service_status.bat' to see if the service is currently running.
   - Right-click 'uninstall_service.bat' and 'Run as administrator' to remove the service.
"""
(DIST_DIR / "README_DEPLOY.txt").write_text(readme_text, encoding="utf-8")
print("Created README_DEPLOY.txt")

print("\n" + "=" * 70)
print("  STANDALONE ONEDIR BUILD SCRIPT COMPLETED SUCCESSFULLY!")
print(f"  Distribution Folder: {DIST_DIR}")
print(f"  Executable:          {DIST_DIR / 'RetailAI.exe'}")
print("=" * 70)
