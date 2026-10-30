@echo off
rem Rifted margins at Earth values (breakup at 10 km of crust, necking zone 100 km; Earth 5-15 km and 50-200 km) with the
rem Earth-calibrated 46 km erosion law and realistic margins (docs/planet-layer-plan.md, seventeenth pass), then the
rem continental crust against CRUST1.0 (scripts\crust_review.py).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion=1.0:200:0.036:True:0.13 --grid breakup_thick_m,rift_width_km=17500:0,10000:100 || goto fail
%PY% -u -m scripts.crust_review --latest || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
