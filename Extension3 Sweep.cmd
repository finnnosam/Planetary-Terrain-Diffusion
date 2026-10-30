@echo off
rem Thirtieth pass (docs/planet-layer-plan.md): E3, extension of thick crust, third version (agreed 25 Oct).
rem  E3 = E2 (twenty-ninth pass) with two changes: (b') ext_clock_flow: the thickening clock is carried by the
rem  collapse flow (a node the flow thickens faster than orogen_strain_rate takes the most recent time of its thicker
rem  timed neighbours; never the present time, so the start's crust stays cold); (c') ext_cut band: the cut ends at
rem  the zone, a microplate between two transforms (small circles about the relative pole on the cut), so the cut no
rem  longer rifts the rest of the plate.
rem  Plus a fix found in the pre-check: ext_pair_motion union, while an event runs the two halves' mean motion
rem  follows the forces on the pair as one plate (a microplate taking the collision boundary no longer speeds its parent).
rem  Combinations: extension False (control; must equal sweep-20261025-111119\extension-False... digit for digit),
rem  E3 onset 45 km, E3 onset 50 km, clock only (ext_cut whole) onset 45 km. 4 x 3 seeds = 12 runs.
rem Decision rule (unchanged from the twenty-seventh pass): gates = the elevation tests pass and the export is
rem 4-8 km3/yr (3-seed means). A candidate is adopted if it passes the gates and at least 2 of the 4 E tests of
rem interior_probe (slope of land above 2 km p50 7.0-17.6 m/km, share under 5 m/km <= 0.35, > 45 km depth from its
rem edge p50 58-146 km and p90 203-510 km) with no guard failing (crust median 34-40 km, young orogen on land above
rem 2 km >= 0.5, abrupt margins <= 30%, edge T 24-29 km, continental volume no worse than -14%, area-ledger residual
rem within 0.005 of the control's). Between candidates that both qualify: more E tests, ties to the smaller > 45 km share.
rem Design checks (not pass bands): interior_probe's "E trigger ... eligible" >= 0.5; rift area over the control no
rem more than the candidate's own extension area (area ledger lines).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\extension3-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\extension3-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,ext_held_rule,ext_hot_myr,ext_clock_flow,ext_pair_motion=1.0:200:0.036:True:0.13:True:30:direction:100:True:union --grid extension,ext_onset_thick_m,ext_cut=False:45000:band,True:45000:band,True:50000:band,True:45000:whole || goto fail
rem the reviews are saved in the new sweep's reviews folder (the newest sweep-* folder) and shown here
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\extension3-pytest.txt "%SW%\reviews\pytest.txt" >nul
for %%r in (elevation_check crust_review margin_width interior_probe) do (
  %PY% -u -m scripts.%%r --latest > "%SW%\reviews\%%r.txt" 2>&1
  type "%SW%\reviews\%%r.txt"
)
echo Reviews saved in %SW%\reviews
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
