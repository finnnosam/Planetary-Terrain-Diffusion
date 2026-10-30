@echo off
rem Thirty-seventh pass (docs/planet-layer-plan.md): item B-b, plates are connected (piece_rule, default 'shared').
rem  'plates': each detached piece of a plate (a connected part other than its largest) with an interior node becomes
rem  its own plate if it covers at least piece_min_area_mkm2 (1 Mkm^2: Bird's 20 plates of at least 1 Mkm^2 hold 98%
rem  of Earth's surface), keeping the parent's Euler vector at the split and then driven by its own boundaries; a
rem  smaller piece joins the plate around most of its edge (microplate capture). Boundary crumbs (no interior node)
rem  are left. sediment_compaction (B-a) is the default since this pass and is passed explicitly here.
rem  Combinations (2 x 3 seeds = 6 runs), the ring control's settings:
rem   piece_rule shared : control; must equal sweep-20261025-201623\sediment_compaction-True field by field
rem                       (stats.json; new keys only: params_effective, detached_area_frac, piece_stats)
rem   piece_rule plates : B-b
rem Decision (fixed before the runs). B-b is adopted if
rem  (a) its tests pass (scripts.plate_pieces_review, "Decision" lines): T1 detached area <= 1% of the surface at
rem      every step on every seed; T2 intra-oceanic trench 12-25k km, T3 continental-side trench 20-35k km, T4 ridge
rem      production 2.5-4.5 km^2/yr, T5 ridge length 94.8k km +- 15%, each as the 3-seed mean of the window -100 ... 0 Myr
rem      (every 10 Myr) and each in its band or no farther from it than the control's. Why the window and "no farther":
rem      one snapshot swings by 10-20k km of trench, and the start alone carries 33-38k km of intra-oceanic trench (its
rem      supercontinent coast is passive; plan doc, thirty-seventh pass), which B-b cannot remove. The plan's forms
rem      (T2 / T3 / T5 at 0 Myr, T4 over the last 40 Myr) are printed above it;
rem  (b) the gates pass (elevation tests on 3 seeds), and
rem  (c) no guard fails: crust median 34-40 km (label-free, 3 seeds), continental volume no worse than the control's
rem      by more than 1 point, margin review as the control's.
rem  The start's convergence is the next design question (plan doc: Pangea's subduction ring). Read beside, not deciding: plate
rem  count and sizes against Bird, the B tests (crust_budget: water below the datum, deep floor, area, sediment
rem  subducted), speeds, run time.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\pieces-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\pieces-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.earth_seafloor_age || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,sediment_compaction=1.0:200:0.036:True:0.13:True:30:True --grid piece_rule=shared,plates || goto fail
rem the reviews are saved in the new sweep's reviews folder (the newest sweep-* folder) and shown here
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\pieces-pytest.txt "%SW%\reviews\pytest.txt" >nul
for %%r in (plate_pieces_review elevation_check crust_review margin_width crust_budget) do (
  %PY% -u -m scripts.%%r --latest > "%SW%\reviews\%%r.txt" 2>&1
  type "%SW%\reviews\%%r.txt"
)
del "%SW%\reviews\trench_origin.txt" 2>nul
del "%SW%\reviews\ocean_sediment_probe.txt" 2>nul
for /d %%c in ("%SW%\*") do if /i not "%%~nxc"=="reviews" (
  %PY% -u -m scripts.trench_origin "%%c" >> "%SW%\reviews\trench_origin.txt" 2>&1
  %PY% -u -m scripts.ocean_sediment_probe "%%c" >> "%SW%\reviews\ocean_sediment_probe.txt" 2>&1
)
type "%SW%\reviews\trench_origin.txt"
type "%SW%\reviews\ocean_sediment_probe.txt"
echo Reviews saved in %SW%\reviews
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
