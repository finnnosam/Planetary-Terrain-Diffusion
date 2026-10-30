@echo off
rem Thirty-fourth pass (docs/planet-layer-plan.md): the Typing E4 sweep (sweep-20261025-170555) made
rem boundary_normal 'ring' the default (PARAMS in planet_layer/plates.py; the default assert in tests/test_plates.py,
rem now test_ring_normal_is_the_default_and_keeps_the_ledgers). All 65 tests passed in the device shell (Linux);
rem this confirms them on Windows.
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
