@echo off
rem Arc docking probe (27 Oct; docs/planet-layer-plan.md, thirty-ninth pass). Gathers information, decides nothing.
rem  Question: why does docking island arcs (B-d, arc_accretion) lengthen the trench and cut plates? On one seed in
rem  the cloud: 34-40 plates vs 20-22, trench about 1.5x the control's from -150 Myr, 300+ small continental bodies.
rem  Runs (3 x 6 seeds = 18 runs, about 1.5 h on this PC), the adopted defaults passed explicitly:
rem   arc_H_ocean_m-300_arc_accretion-False  : control (today's model); must equal piece_rule-plates of
rem                                            sweep-20261025-220434 (1001-1003) and sweep-20261026-204140 (1004-1006)
rem   arc_H_ocean_m-1000_arc_accretion-False : arc rate from Earth alone
rem   arc_H_ocean_m-1000_arc_accretion-True  : rate and docking
rem  Runs now also save every plate event with its time (stats.json event_list; output only).
rem  Reviews in the new sweep's reviews folder:
rem   identity.txt         control vs the earlier runs, field by field (should say IDENTICAL twice)
rem   arc_dock_probe.txt   when the trench parts from the control's, plate events by 50 Myr window, where the trench
rem                        sits (near small continental bodies / thin continent / oceanic arcs / large continents / far),
rem                        the small bodies' size and thickness, spikes, the plates that cut others, arc fluxes;
rem                        plus arc_dock_probe.png (trench and plates against time, per seed) and .json
rem   plate_pieces_review, crust_budget (arc tests A1-A3), elevation_check, crust_review, margin_width as usual
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\arc-probe-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\arc-probe-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003,1004,1005,1006 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,sediment_compaction,piece_rule=1.0:200:0.036:True:0.13:True:30:True:plates --grid arc_H_ocean_m,arc_accretion=300:False,1000:False,1000:True || goto fail
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\arc-probe-pytest.txt "%SW%\reviews\pytest.txt" >nul
set CTRL=%SW%\arc_H_ocean_m-300_arc_accretion-False
%PY% tools\compare_runs.py "%CTRL%" %CAL%\sweep-20261025-220434\piece_rule-plates > "%SW%\reviews\identity.txt" 2>&1
%PY% tools\compare_runs.py "%CTRL%" %CAL%\sweep-20261026-204140\piece_rule-plates >> "%SW%\reviews\identity.txt" 2>&1
type "%SW%\reviews\identity.txt"
%PY% -u -m scripts.arc_dock_probe "%SW%" > "%SW%\reviews\arc_dock_probe.txt" 2>&1
type "%SW%\reviews\arc_dock_probe.txt"
%PY% -u -m scripts.plate_pieces_review --latest --control arc_H_ocean_m-300_arc_accretion-False > "%SW%\reviews\plate_pieces_review.txt" 2>&1
type "%SW%\reviews\plate_pieces_review.txt"
for %%r in (crust_budget elevation_check crust_review margin_width) do (
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
