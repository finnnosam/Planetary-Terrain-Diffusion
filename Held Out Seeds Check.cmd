@echo off
rem Held-out seeds check (26 Oct night; docs/planet-layer-plan.md, "Investigation ... dynamic-topography field").
rem  Every sweep since 17 Oct was decided on seeds 1001-1003, and the dynamic-topography field (Mantle, its own stream
rem  default_rng([seed, 7])) is identical in every run of a seed: all adopted values were chosen against the same three
rem  mantle histories. This runs the current baseline on seeds never used before (1004, 1005, 1006), no code change.
rem  Combinations (2 x 3 seeds = 6 runs), the same overrides as Coast Start Sweep.cmd:
rem   start_plates-voronoi_start_poles-random : the baseline (control of sweep-20261026-184745)
rem   start_plates-coast_start_poles-forces   : the Coast Start candidate, so its decision is checked on new seeds too
rem  Reference: sweep-20261026-184745, the same two combinations on seeds 1001-1003.
rem Rule (fixed before the runs):
rem  (1) Adopted tests: the gates (elevation tests) and guards (crust median 34-40 km, margin review) that passed on
rem      1001-1003 must pass on 1004-1006. A gate that fails here reopens the adoption it came from.
rem  (2) Headline numbers (> 45 km share, land > 2 km share, young-ocean offset, water below the datum, continental
rem      volume change, crust median, T2-T5 window means): the held-out 3-seed mean is flagged if it lies outside the
rem      reference's seed min-max widened by half that range on each side. Two or more flags = the old seeds were not
rem      representative: decide again on all six seeds.
rem  (3) Coast Start: if coast:forces is adopted from the reference sweep, it must still pass its own rule against the
rem      control on 1004-1006 (start S1 / S2, T1-T5, gates, guards).
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1004,1005,1006 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,sediment_compaction,piece_rule=1.0:200:0.036:True:0.13:True:30:True:plates --grid start_plates,start_poles=voronoi:random,coast:forces || goto fail
rem the reviews are saved in the new sweep's reviews folder (the newest sweep-* folder) and shown here
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
echo held-out seeds 1004-1006; reference sweep-20261026-184745 > "%SW%\reviews\README.txt"
%PY% -u -m scripts.plate_pieces_review --latest --control start_plates-voronoi_start_poles-random > "%SW%\reviews\plate_pieces_review.txt" 2>&1
type "%SW%\reviews\plate_pieces_review.txt"
for %%r in (elevation_check crust_review margin_width crust_budget) do (
  %PY% -u -m scripts.%%r --latest > "%SW%\reviews\%%r.txt" 2>&1
  type "%SW%\reviews\%%r.txt"
)
for /d %%c in ("%SW%\*") do if /i not "%%~nxc"=="reviews" (
  %PY% -u -m scripts.trench_origin "%%c" >> "%SW%\reviews\trench_origin.txt" 2>&1
)
type "%SW%\reviews\trench_origin.txt"
echo Reviews saved in %SW%\reviews
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
