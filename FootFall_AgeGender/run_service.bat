@echo off
REM Supervisor loop: restarts the app if it ever exits. Register with Task Scheduler (At startup).
cd /d "%~dp0"
REM Secrets live here or in system environment variables - not in git.
if exist "%~dp0edge.env.bat" call "%~dp0edge.env.bat"
:loop
call venv\Scripts\activate.bat
python app.py --no-display --no-demographics --config config/production.yaml %*
echo [%date% %time%] app exited with code %ERRORLEVEL%, restarting in 5s >> restarts.log
timeout /t 5 /nobreak >nul
goto loop
