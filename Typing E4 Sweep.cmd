@echo off
rem Thirty-third pass (docs/planet-layer-plan.md): boundary typing (boundary_normal ring) and E4 (ext_side free).
rem  Boundary typing: classify_boundaries typed each node by the one neighbour with the largest normal rate (a grid
rem  direction); on the grid's staircase a transform that does not follow the grid was typed ridge / collision /
rem  subduction (pure transforms: median 30% of nodes typed transform, min 0%). boundary_normal ring takes the normal
rem  from the node's two-ring (96-100% typed transform). Found because E3/E4 microplates docked onto their own
rem  parent across the band's side edges (transforms typed as collision) after exactly merge_after_myr.
rem  E4: ext_side free, at a band cut the side whose far edge faces fewer collision nodes moves (Rey et al. 2001).
rem  Combinations (5 x 3 seeds = 15 runs):
rem   node, extension False  : control; must equal sweep-20261025-141028\extension-False... digit for digit
rem   ring, extension False  : the typing fix alone
rem   ring, E4 45 km / 50 km : E4 on the fixed typing
rem   node, E4 50 km         : E4 without the fix (the pre-check's case on your machine)
rem Decision, typing (a correctness fix, read first): ring is adopted as the default if its control passes the
rem gates (elevation tests, export 4-8 km3/yr) and the guards (crust median 34-40 km, continental volume no worse than
rem -14%); read beside it: boundary lengths by type vs Bird 2003 (boundary_lengths.txt), plate count and speed,
rem young-ocean offset. A gate that fails with ring points to the next missing mechanism: stop and read, do not
rem revert to pass it.
rem Decision, E4 (unchanged rule, twenty-seventh pass): gates; at least 2 of the 4 E tests of interior_probe (slope
rem of land above 2 km p50 7.0-17.6 m/km, share under 5 m/km <= 0.35, > 45 km depth from its edge p50 58-146 km and
rem p90 203-510 km) with no guard failing (crust median 34-40 km, young orogen on land above 2 km >= 0.5, abrupt
rem margins <= 30%, edge T 24-29 km, continental volume no worse than -14%, area-ledger residual within 0.005 of its
rem control's), compared with the control on the same typing. Design checks: E trigger eligible >= 0.5; rift area over
rem the control <= the candidate's own extension area; lost events <= 1/3 of starts and far side on a collision
rem median <= 0.2 ("E4 design checks" line).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\typing-e4-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\typing-e4-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,ext_held_rule,ext_hot_myr,ext_clock_flow,ext_pair_motion,ext_cut,ext_side=1.0:200:0.036:True:0.13:True:30:direction:100:True:union:band:free --grid boundary_normal,extension,ext_onset_thick_m=node:False:45000,ring:False:45000,ring:True:45000,ring:True:50000,node:True:50000 || goto fail
rem the reviews are saved in the new sweep's reviews folder (the newest sweep-* folder) and shown here
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\typing-e4-pytest.txt "%SW%\reviews\pytest.txt" >nul
for %%r in (elevation_check crust_review margin_width interior_probe) do (
  %PY% -u -m scripts.%%r --latest > "%SW%\reviews\%%r.txt" 2>&1
  type "%SW%\reviews\%%r.txt"
)
rem boundary lengths by type (contour-typed and node-counted) of every run's present snapshot
del "%SW%\reviews\boundary_lengths.txt" 2>nul
for /r "%SW%" %%f in (t+0000.0.npz) do if exist "%%f" (
  echo %%f>> "%SW%\reviews\boundary_lengths.txt"
  %PY% -m scripts.boundary_lengths "%%f" >> "%SW%\reviews\boundary_lengths.txt" 2>&1
)
type "%SW%\reviews\boundary_lengths.txt"
echo Reviews saved in %SW%\reviews
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
