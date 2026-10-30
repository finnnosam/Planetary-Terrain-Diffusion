@echo off
rem Twentieth pass (docs/planet-layer-plan.md): what thickens old passive margins into abrupt edges.
rem  1. continental_sediment: sediment on continental crust as its own layer (density 2,400, a flexural load;
rem     erosion takes sediment first). Before, deposits on continents were booked as crust of density 2,830.
rem  2. crust_kappa_normal_m2_yr 30 instead of 1000: lateral flow of normal-thickness crust from a lower-crust
rem     viscosity of ~1e21 Pa s (1000 is ~4e19, hot crust); 180 Myr old Atlantic margins survive, which needs
rem     kappa well below ~125 m2/yr. The collapse flow of thick crust (above 45 km) is unchanged.
rem Earth erosion law, realistic margins, Earth rifted margins (10 km / 100 km), local isostasy on (default);
rem three settings: neither, the layer, the layer + kappa 30; 3 seeds. Then the elevation tests, the crust, the
rem margin width, the margin age and the margin tracking.
rem Decision rule (fixed before the sweep): adopt each Earth value whose 3-seed means pass the elevation tests;
rem every other band is read as it comes out (no band tuning).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,breakup_thick_m,rift_width_km,continental_sediment,crust_kappa_normal_m2_yr=1.0:200:0.036:True:0.13:10000:100:False:1000,1.0:200:0.036:True:0.13:10000:100:True:1000,1.0:200:0.036:True:0.13:10000:100:True:30 || goto fail
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
