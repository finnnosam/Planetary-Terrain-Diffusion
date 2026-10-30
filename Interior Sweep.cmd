@echo off
rem Twenty-fourth pass (docs/planet-layer-plan.md): the thick interior (item 2).
rem  start_crust  assumed  (until now): the supercontinent starts at ~40 km (37.5 + 2 inland + 4 near sutures).
rem               earth    (A): the same field quantile-mapped onto Earth's stable crust (CRUST1.0 Precambrian and
rem                        platform types > 500 km from plate boundaries, crystalline p10 / 50 / 90 = 32.9 / 38.2 / 43.0).
rem  orogen_rule  collision (until now): only collision contacts reset the orogeny clock (erodibility 1).
rem               belt     (B): also the subduction belt (d0 +- 2 sigma, converging) and crust thicker than 45 km.
rem               strain   (B, measured): also crust deforming faster than 3.5e-9 /yr (|thickening| / thickness), the
rem                        GSRM v2.1 areal dilatation that flags as much of Earth's land as Bird's orogens (17%).
rem Baseline otherwise: Earth erosion law, realistic margins, pure-shear rifting (10 km / 100 km, defaults), local
rem isostasy, continental sediment layer, kappa_normal 30; 3 seeds; 6 combinations x 3 seeds = 18 runs (~1 h).
rem Then the elevation tests and export, the crust, the margins and the interior.
rem Decision rule (fixed before the sweep): gates = the elevation tests pass and the export is 4-8 km3/yr (3-seed
rem means, elevation_check). Among the combinations through the gates, the one passing most of the five item-2
rem tests of interior_probe (inland profile 150-1000 km within 1.5 km of Earth's; > 45 km <= 0.13; young-orogen
rem erodibility on 0.10-0.25 of land and on >= 0.5 of land above 2 km; land above 2 km <= 0.10) plus the crust
rem median 34-40 km (crust_review) wins; ties go to the smaller > 45 km share. Its start_crust and orogen_rule become
rem the defaults. If no combination passes the gates, nothing is adopted. Every other band is read as it comes out.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr=1.0:200:0.036:True:0.13:True:30 --grid start_crust=assumed,earth --grid orogen_rule=collision,belt,strain || goto fail
%PY% -u -m scripts.elevation_check --latest || goto fail
%PY% -u -m scripts.crust_review --latest || goto fail
%PY% -u -m scripts.margin_width --latest || goto fail
%PY% -u -m scripts.interior_probe --latest || goto fail
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
