@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Install the project virtual environment first. See README.md.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m planet_diffusion.gui
