@echo off
rem Thirty-eighth pass (docs/planet-layer-plan.md): the start's plate geometry and motion.
rem  start_plates (default 'voronoi'): 'coast' grows the plate seeds only within their own domain, so the
rem  supercontinent's plates end at its coast and the oceanic plates fill the ocean; the rim is a plate boundary all
rem  round. 'voronoi': cells of the whole sphere; the supercontinent's plates reach into the ocean, its coast is
rem  passive and all closing is ocean-ocean (continental-side trench 0.2-0.7k km at -250 Myr, intra-oceanic 33-38k).
rem  start_poles (default 'random'): 'forces' gives each oceanic start plate the speed and pole of the plate layer's own
rem  force law on the start's geometry, with its coast nodes as trench (Pangea's subduction ring: Collins 2003; Cawood
rem  2005) and ocean-ocean boundaries as ridges; continental plates keep their dispersal motion. No new numbers.
rem  piece_rule 'plates' (B-b) is the default since this pass and is passed explicitly here. The ledgers no longer
rem  drift (consumed nodes booked when they leave; no double count of unread collision volume): ledgers only.
rem  Combinations (3 x 3 seeds = 9 runs), the B-b settings:
rem   start_plates-voronoi_start_poles-random : control; must equal sweep-20261025-220434\piece_rule-plates field by field
rem                                             (stats.json; only the overrides, params_effective, run time and the
rem                                             ledger keys collision_out / cont_subducted / cont_lost_other may differ)
rem   start_plates-coast_start_poles-random   : the ringed start, random oceanic poles
rem   start_plates-coast_start_poles-forces   : the ringed start, oceanic poles from the force law
rem Decision (fixed before the runs). A candidate passes if
rem  (a) the start passes (the "start" line under its Decision line in plate_pieces_review.txt, 3-seed means at
rem      -240 Myr): S1 continental-side trench >= 20k km and S2 its share of all subduction >= 0.60 (Earth today 0.60);
rem  (b) over the run it is no worse: T1 detached area <= 1% at every step on every seed, and T2-T5 as 3-seed window
rem      means (-100 ... 0 Myr, every 10 Myr) each in its band or no farther from it than the control's;
rem  (c) the gates pass (elevation tests on 3 seeds), and
rem  (d) no guard fails: crust median 34-40 km (label-free, 3 seeds), continental volume no worse than the control's by
rem      more than 1 point, margin review as the control's.
rem  If both pass, the one with more of T2-T5 strictly inside their bands is adopted; on a tie, the smaller window T2.
rem  Read beside, not deciding: the B tests (crust_budget), plate count and sizes, speeds, ocean-floor age distribution
rem  against Seton et al. 2020, run time.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\coast-start-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\coast-start-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,sediment_compaction,piece_rule=1.0:200:0.036:True:0.13:True:30:True:plates --grid start_plates,start_poles=voronoi:random,coast:random,coast:forces || goto fail
rem the reviews are saved in the new sweep's reviews folder (the newest sweep-* folder) and shown here
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\coast-start-pytest.txt "%SW%\reviews\pytest.txt" >nul
%PY% -u -m scripts.plate_pieces_review --latest --control start_plates-voronoi_start_poles-random > "%SW%\reviews\plate_pieces_review.txt" 2>&1
type "%SW%\reviews\plate_pieces_review.txt"
for %%r in (elevation_check crust_review margin_width crust_budget) do (
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
