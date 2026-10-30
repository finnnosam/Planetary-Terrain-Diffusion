@echo off
rem Twenty-ninth pass (docs/planet-layer-plan.md): E2, extension of thick crust, revised (agreed 25 Oct).
rem  E2 = E (twenty-seventh pass) with three changes: (a) no held exemption: a thick piece touching a collision
rem  boundary or subduction belt is cut ACROSS its long axis (orogen-parallel stretching, Tibet), a free piece along
rem  it (Basin and Range, Aegean); (b) only hot crust extends: more than half of the piece thickened tectonically
rem  (collision volume, margin shortening, arc/subduction thickening; not collapse diffusion) within ext_hot_myr
rem  = 100 Myr (Sonder et al. 1987), so the start's crust never triggers; (c) the cut stays whole-plate.
rem  extension False: the current defaults (control; must equal sweep-20261024-215311\extension-False... digit for digit).
rem Baseline otherwise as the extension sweep. 3 x 3 seeds = 9 runs.
rem Decision rule (unchanged from the twenty-seventh pass): gates = the elevation tests pass and the export is
rem 4-8 km3/yr (3-seed means). E2 is adopted if it passes the gates and at least 2 of the 4 E tests of
rem interior_probe (slope of land above 2 km p50 7.0-17.6 m/km, share under 5 m/km <= 0.35, > 45 km depth from its
rem edge p50 58-146 km and p90 203-510 km) with no guard failing (crust median 34-40 km, young orogen on land above
rem 2 km >= 0.5, abrupt margins <= 30%, edge T 24-29 km, continental volume no worse than -14%, area-ledger residual
rem within 0.005 of the control's). Between onsets that both qualify: more E tests, ties to the smaller > 45 km share.
rem Design check (not a pass band): interior_probe's "E trigger ... eligible" >= 0.5.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\extension2-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\extension2-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,ext_held_rule,ext_hot_myr=1.0:200:0.036:True:0.13:True:30:direction:100 --grid extension,ext_onset_thick_m=False:45000,True:45000,True:50000 || goto fail
rem the reviews are saved in the new sweep's reviews folder (the newest sweep-* folder) and shown here
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\extension2-pytest.txt "%SW%\reviews\pytest.txt" >nul
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
