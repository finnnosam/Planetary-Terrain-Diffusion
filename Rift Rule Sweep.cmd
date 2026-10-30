@echo off
rem Twenty-first pass (docs/planet-layer-plan.md): a rift rule stated in distance.
rem  rift_rule node        (until now): a rift gap gets 0.6 x its continental neighbours' thickness, a factor per
rem                        grid cell: ~2 cells of stretched crust per side, 45-70 km of thin crust per km of edge.
rem  rift_rule pure_shear  (new): the gap is the new area of a stretching zone of half-width rift_width_km (100 km,
rem                        Earth's necking zone, the value already in use), linear taper; the zone thins as pure
rem                        shear and the gap gets exactly the volume the zone loses; breakup below 10 km.
rem Baseline otherwise: Earth erosion law, realistic margins, Earth rifting (10 km / 100 km), local isostasy,
rem continental sediment layer, kappa_normal 30; 3 seeds. Then the elevation tests, the crust, the margin width
rem (now also read through CRUST1.0's 1 degree cells), the margin age and the margin tracking.
rem Decision rule (fixed before the sweep): adopt pure_shear if its 3-seed means pass the elevation tests and its
rem thin crust per km of passive edge (1 degree reading) is inside 100-200 km; every other band is read as it
rem comes out (no band tuning).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,breakup_thick_m,rift_width_km,continental_sediment,crust_kappa_normal_m2_yr=1.0:200:0.036:True:0.13:10000:100:True:30 --grid rift_rule=node,pure_shear || goto fail
%PY% -u -m scripts.elevation_check --latest || goto fail
%PY% -u -m scripts.crust_review --latest || goto fail
%PY% -u -m scripts.margin_width --latest || goto fail
%PY% -u -m scripts.margin_age_probe --latest || goto fail
%PY% -u -m scripts.margin_track_probe --latest --from -100 || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
