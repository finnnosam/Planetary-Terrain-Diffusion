@echo off
rem Eighteenth pass: ocean-floor erosion now charged to the oceanic crust (ledger closed, always on), and thinned
rem continental crust compensated locally (thinning_local_isostasy). Earth erosion law, realistic margins, Earth rifted
rem margins (10 km / 100 km); with and without the local isostasy, 3 seeds. Then the elevation tests and the crust.
rem The same setting with the old code is sweep-20261023-225025 (docs/planet-layer-plan.md, eighteenth pass).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,breakup_thick_m,rift_width_km,thinning_local_isostasy=1.0:200:0.036:True:0.13:10000:100:False,1.0:200:0.036:True:0.13:10000:100:True || goto fail
%PY% -u -m scripts.elevation_check --latest || goto fail
%PY% -u -m scripts.crust_review --latest || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
