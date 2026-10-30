@echo off
rem Erosion laws of the planet chain against measured 10Be erosion rates (docs/planet-layer-plan.md, step 8 follow-up 1,
rem fourteenth pass). Needs data\octopus\crn_int_basins.json (OCTOPUS v2 CRN Global basins, GeoJSON, EPSG:4326).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
if not exist data\octopus\crn_int_basins.json (echo data\octopus\crn_int_basins.json is missing & goto fail)
%PY% -m pytest -q tests\test_erosion_law_check.py || goto fail
%PY% -u -m scripts.erosion_law_check --grid 217 || goto fail
%PY% -u -m scripts.erosion_law_check --grid 868 || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
