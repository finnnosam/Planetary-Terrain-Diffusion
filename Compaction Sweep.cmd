@echo off
rem Thirty-sixth pass (docs/planet-layer-plan.md): item B-a, sediment compaction (sediment_compaction, default False).
rem  With it the sediment field is solid (grain) volume, rho_grain 2650, and the elevation reads the bulk height and
rem  mass of a column compacting by Athy's law (Sclater & Christie 1980: shale phi0 0.63, c 0.51 /km on oceanic crust;
rem  shaly sandstone 0.56, 0.39 /km on continental crust; water-filled pores). Ledgers stay in solid volume; the
rem  landscape kernels are unchanged (the 11.53 km stage is untouched). Snapshots gain sediment_bulk_m; the reviews
rem  read it where they compare with CRUST1.0 or GlobSed.
rem  Combinations (2 x 3 seeds = 6 runs), the ring control's settings of sweep-20261025-170555:
rem   sediment_compaction False : control; must equal sweep-20261025-170555\boundary_normal-ring_extension-False...
rem                               digit for digit (stats.json)
rem   sediment_compaction True  : B-a
rem Decision (fixed before the runs). The floor test is paired, on the same states (runs drift apart from the first
rem steps, so a sweep sees B-a's floor only through seed spread): scripts.compaction_effect on sweep-20261025-170555's
rem ring control gave the deep floor (mean depth of the floor deeper than 1.5 km below the datum) -71 / -115 / -171 m
rem at 0 Myr (mean -119: passes "toward Earth's 4,205 m by at least 100 m without passing 4,055 m"); it is repeated
rem here on this sweep's control. The sweep decides on the dynamics: B-a is adopted if the gates pass (elevation
rem tests on 3 seeds) with no guard failing (crust median 34-40 km, continental volume no worse than the control's by
rem more than 1 point, margin review as the control's). Read beside it, not deciding: water below the datum (Earth
rem 2,629 m), sea level on the datum, the ocean's bulk sediment against GlobSed (ocean_sediment_probe: label-free
rem ocean mean 610 m), land above 2 km, export.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\compaction-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\compaction-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.earth_sediment || goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr=1.0:200:0.036:True:0.13:True:30 --grid sediment_compaction=False,True || goto fail
rem the reviews are saved in the new sweep's reviews folder (the newest sweep-* folder) and shown here
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\compaction-pytest.txt "%SW%\reviews\pytest.txt" >nul
for %%r in (elevation_check crust_review margin_width crust_budget) do (
  %PY% -u -m scripts.%%r --latest > "%SW%\reviews\%%r.txt" 2>&1
  type "%SW%\reviews\%%r.txt"
)
%PY% -u -m scripts.compaction_effect "%SW%\sediment_compaction-False" --times 0,-20 > "%SW%\reviews\compaction_effect.txt" 2>&1
type "%SW%\reviews\compaction_effect.txt"
del "%SW%\reviews\ocean_sediment_probe.txt" 2>nul
for /d %%c in ("%SW%\*") do if /i not "%%~nxc"=="reviews" (
  %PY% -u -m scripts.ocean_sediment_probe "%%c" >> "%SW%\reviews\ocean_sediment_probe.txt" 2>&1
)
type "%SW%\reviews\ocean_sediment_probe.txt"
echo Reviews saved in %SW%\reviews
echo done
pause
exit /b 0
:fail
echo FAILED, see the messages above
pause
exit /b 1
