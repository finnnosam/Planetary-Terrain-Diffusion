@echo off
rem Collapse of thick crust past the continent edge, with Earth rifted margins (breakup 10 km, necking 100 km), the
rem Earth-calibrated 46 km erosion law and realistic margins (docs/planet-layer-plan.md, seventeenth pass). Then the
rem continental crust against CRUST1.0. The same setting without edge collapse is sweep-20261023-225025.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_plates.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,breakup_thick_m,rift_width_km,collapse_edge=1.0:200:0.036:True:0.13:10000:100:True || goto fail
%PY% -u -m scripts.crust_review --latest || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
