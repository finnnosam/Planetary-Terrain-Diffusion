@echo off
rem B-b on six seeds (26 Oct night; docs/planet-layer-plan.md, "Held-out seeds check read").
rem  The held-out check (sweep-20261026-200857, seeds 1004-1006) flagged three of B-b's window tests for the baseline
rem  (T2 intra-oceanic trench, T4 production, T5 ridge: all outside the 1001-1003 range widened by half), so by its rule
rem  B-b (piece_rule 'plates', adopted on 1001-1003 in sweep-20261025-220434) is decided again on six seeds.
rem  Combinations (2 x 3 seeds = 6 runs), the same overrides as the held-out check, default start (voronoi:random):
rem   piece_rule-plates : must equal sweep-20261026-200857\start_plates-voronoi_start_poles-random field by field
rem                       (runs are deterministic on this machine; only params_effective and run time may differ)
rem   piece_rule-shared : B-b's control on the new seeds
rem Rule (fixed before the runs), on the six-seed means (1001-1003 from sweep-20261025-220434, 1004-1006 from here):
rem  B-b stays adopted if T1 passes on every seed and each of T2-T5 window means (-100 ... 0 Myr) is in its band or no
rem  farther from it than shared's, the gates pass on at least 5 of 6 seeds for plates and no fewer than for shared,
rem  and no guard fails (crust median 34-40 km, volume no worse than shared's by more than 1 point, margins as shared's).
rem  Otherwise piece_rule goes back to 'shared' and the plan doc records why.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1004,1005,1006 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,sediment_compaction=1.0:200:0.036:True:0.13:True:30:True --grid piece_rule=plates,shared || goto fail
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
echo B-b on held-out seeds 1004-1006; reference sweep-20261025-220434 (1001-1003) > "%SW%\reviews\README.txt"
%PY% -u -m scripts.plate_pieces_review --latest --control piece_rule-shared > "%SW%\reviews\plate_pieces_review.txt" 2>&1
type "%SW%\reviews\plate_pieces_review.txt"
for %%r in (elevation_check crust_review margin_width crust_budget) do (
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
