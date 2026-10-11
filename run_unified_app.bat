@echo off
title OmniRetail AI Suite - Customer Analytics and VisionGuard Staff Attendance
color 0b
echo ===============================================================================
echo     OmniRetail AI Suite - Unified Desktop Edge Intelligence Gateway
echo     Customer Analytics (FootFall and Demographics) + VisionGuard AI (Attendance)
echo ===============================================================================
echo.
cd /d "%~dp0"

set OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp
set CUDA_VISIBLE_DEVICES=-1
set YOLO_AUTOINSTALL=0
set PYTHONUNBUFFERED=1

echo Starting Unified Desktop GUI Application...
python unified_app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with code %ERRORLEVEL%.
    pause
)
