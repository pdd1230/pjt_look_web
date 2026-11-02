@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip check
if errorlevel 1 goto failed
if not exist .env copy .env.example .env >nul
echo Setup complete. Edit .env and configure MySQL before start_windows.bat.
pause
exit /b 0
:failed
echo Setup failed. Read the error above. Python 3.11 must be installed.
pause
exit /b 1
