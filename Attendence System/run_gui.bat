@echo off
title VisionGuard AI — Real-Time Face Attendance
cd /d "%~dp0"
echo ========================================================
echo  Launching VisionGuard AI Windows Native Desktop GUI
echo ========================================================
python main_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error. Press any key to exit.
    pause
)
