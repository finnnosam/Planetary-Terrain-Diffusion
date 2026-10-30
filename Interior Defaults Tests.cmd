@echo off
rem Twenty-fifth pass (docs/planet-layer-plan.md): the interior sweep's rule made start_crust 'earth' and
rem orogen_rule 'belt' the defaults (PARAMS in planet_layer/plates.py, two asserts in tests/test_plates.py).
rem The planet-layer tests could not run in the device shell that session (no numba / ducc0); run them here.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
