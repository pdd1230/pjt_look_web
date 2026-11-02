@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
if not exist .env (
  echo Copy .env.example to .env and configure MySQL first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\check_environment.py
if errorlevel 1 (
  echo Environment check failed. Check MySQL or model configuration above.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
pause
