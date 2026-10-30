@echo off
rem Plate layer with the 46 km erosion law calibrated on Earth measured 10Be rates (docs/planet-layer-plan.md, step 8
rem follow-up 1, fifteenth pass), crossed with the margin modes. 3 seeds, shelf gate; the current baseline is listed first.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_plates.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc=0.75:6:0.05,1.0:200:0.036,0.1:100:0.036 --grid margin_shortening,sediment_accretion=False:0.75,True:0.13 || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
