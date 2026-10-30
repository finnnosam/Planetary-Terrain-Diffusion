@echo off
rem Trench Flux Sweep (28 Oct; docs/planet-layer-plan.md, fortieth pass). Decides by the rule below, on six seeds.
rem  Found by the Arc Dock Probe (sweep-20261027-210209) and its follow-up:
rem   (1) the trench-flux law (arcs, subduction erosion) adds closing rate x H along node-typed trenches, which on the
rem       planet's rough boundaries run 1.2-2.7x the true line: island arcs got 1.8-2.7x H x the consumed area (on three
rem       of six seeds more arc crust than the whole planet's subduction consumed). trench_flux consumed: each trench term
rem       is H x the oceanic area the trench consumed, spread over the overriding trench node's band.
rem   (2) the trench (downgoing) side of an ocean-ocean subduction node is set by node age, while the overlap resolution
rem       consumes crust by one polarity per plate pair; they disagree at ~22% of oceanic overriding nodes (arcs on the
rem       downgoing plate, slab pull on the overriding one). trench_polarity pair: nodes follow the pair's polarity.
rem  Arms (3 x 6 seeds = 18 runs, about 1.5 h), the adopted defaults passed explicitly:
rem   A trench_polarity-node_trench_flux-band_arc_H_ocean_m-300_arc_accretion-False : control (today's model); must equal
rem       arc_H_ocean_m-300_arc_accretion-False of sweep-20261027-210209 field by field (identity.txt: IDENTICAL)
rem   B trench_polarity-pair_trench_flux-band_arc_H_ocean_m-300_arc_accretion-False  : the polarity fix alone
rem   C trench_polarity-pair_trench_flux-consumed_arc_H_ocean_m-1000_arc_accretion-True : polarity, fluxes from consumed
rem       area, Earth's arc rate (Dimalanta et al. 2002 over Bird's intra-oceanic closing rate) and docking (B-d)
rem  Rule (fixed 28 Oct before the runs), six-seed means unless stated:
rem   guards for B and C against A: T1 pass; T2-T5 window means (-100..0 Myr) each in band or no farther than A's
rem   (plate_pieces_review "Decision"); elevation gates pass on >= 5 of 6 seeds and on no fewer than A; label-free crust
rem   median 34-40 km; continental volume no worse than A's by more than 1 point; margin review passes if A's passes.
rem   C also: routed share >= 0.9 (crust_budget "Trench fluxes"); A1, A2, A3 pass; each of the B tests (water below the
rem   datum, deep floor, volume, label-free area) no farther from Earth than A's; > 45 km share no more than A's + 0.02.
rem   C adopted if it passes (PARAMS: trench_polarity pair, trench_flux consumed, arc_H_ocean_m 1000, arc_accretion True);
rem   else B adopted if it passes (trench_polarity pair); else neither.
cd /d "%~dp0"
set PY=".venv\Scripts\python.exe"
set CAL=outputs\planet-calibration
%PY% -m pytest -q tests\test_shelf.py tests\test_plates.py tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py tests\test_erosion_law_check.py tests\test_crust_reviews.py > %CAL%\trench-flux-pytest.txt 2>&1
set RC=%errorlevel%
type %CAL%\trench-flux-pytest.txt
if not "%RC%"=="0" goto fail
%PY% -u -m scripts.sweep_plates --rule shelf --seeds 1001,1002,1003,1004,1005,1006 --grid erosion_K_factor,erosion_theta,erosion_Sc,margin_shortening,sediment_accretion,continental_sediment,crust_kappa_normal_m2_yr,sediment_compaction,piece_rule=1.0:200:0.036:True:0.13:True:30:True:plates --grid trench_polarity,trench_flux,arc_H_ocean_m,arc_accretion=node:band:300:False,pair:band:300:False,pair:consumed:1000:True || goto fail
for /f "delims=" %%d in ('dir /b /ad /o-n %CAL%\sweep-*') do (set "SW=%CAL%\%%d" & goto found)
:found
mkdir "%SW%\reviews" 2>nul
copy /y %CAL%\trench-flux-pytest.txt "%SW%\reviews\pytest.txt" >nul
set CTRL=trench_polarity-node_trench_flux-band_arc_H_ocean_m-300_arc_accretion-False
%PY% tools\compare_runs.py "%SW%\%CTRL%" %CAL%\sweep-20261027-210209\arc_H_ocean_m-300_arc_accretion-False > "%SW%\reviews\identity.txt" 2>&1
type "%SW%\reviews\identity.txt"
%PY% -u -m scripts.plate_pieces_review --latest --control %CTRL% > "%SW%\reviews\plate_pieces_review.txt" 2>&1
type "%SW%\reviews\plate_pieces_review.txt"
%PY% -u -m scripts.arc_dock_probe "%SW%" --control %CTRL% > "%SW%\reviews\arc_dock_probe.txt" 2>&1
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
