@echo off
cd /d "%~dp0"
echo Generating with the corrected sampler and your existing 222000-step checkpoint.
echo Output: outputs\meso\planet-fixed\planet.tif
echo.
".venv\Scripts\python.exe" -m toy_meso.planet_tiled --run outputs\meso\run2-lowlr --dem human_input\B2W.png --black-metres -6000 --white-metres 6000 --out outputs\meso\planet-fixed --steps 32 --batch 16
if errorlevel 1 echo Generation stopped. See the message above for details.
pause
