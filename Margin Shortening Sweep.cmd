@echo off
rem Margin source: volume-conserving upper-plate shortening + Earth-scaled arc addition, and the trench accretion share
rem (docs/planet-layer-plan.md, step 8 follow-up 1, eleventh pass). 3-seed sweep with the shelf gate; the baseline is
rem listed first, so a new mode wins only if it passes the gate and scores more than 0.03 better.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid margin_shortening,sediment_accretion=False:0.75,True:0.13,True:0.75 || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
