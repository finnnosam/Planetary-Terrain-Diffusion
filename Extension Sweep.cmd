@echo off
rem Twenty-seventh pass (docs/planet-layer-plan.md): E, collapse of thick crust as continental extension.
rem  extension False: the current defaults (control; should reproduce sweep-20261024-193840's earth / belt runs).
rem  extension True : thick crust (> ext_onset_thick_m) that no collision boundary or subduction belt holds splits its
rem                   plate along the piece's long axis; the halves diverge at 2e-8 /yr x the zone's width, the gaps
rem                   fill by pure shear within the zone's half-width (100-400 km), and the halves weld when the
rem                   extended crust's mean reaches 30 km. Onset 45 or 50 km (Earth: 45-50 km before extension).
rem Baseline otherwise as the interior sweep (Earth erosion law, realistic margins, pure-shear rifting, local isostasy,
rem sediment layer, kappa_normal 30; start_crust earth and orogen_rule belt are defaults). 3 x 3 seeds = 9 runs.
rem Decision rule (fixed before the sweep, plan doc twenty-seventh pass): gates = the elevation tests pass and the
rem export is 4-8 km3/yr (3-seed means). E is adopted if it passes the gates and at least 2 of the 4 E tests of
rem interior_probe (slope of land above 2 km p50 7.0-17.6 m/km, share under 5 m/km <= 0.35, > 45 km depth from its
rem edge p50 58-146 km and p90 203-510 km; the control passes 0) with no guard failing (crust median 34-40 km, young
rem orogen on land above 2 km >= 0.5, abrupt margins <= 30%, edge T 24-29 km, continental volume no worse than -14%,
rem area-ledger residual within 0.005 of the control's).
rem Between onsets that both qualify: more E tests, ties to the smaller > 45 km share.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr=1.0:200:0.036:True:0.13:True:30 --grid extension,ext_onset_thick_m=False:45000,True:45000,True:50000 || goto fail
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
