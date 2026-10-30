# Next chat: planet layer

Project: E:\projects\Infinite_Diffusion (Windows, PowerShell, `.venv` Python 3.12, numba, RTX 3070 8 GB, 16 threads).

**Read first, in this order:**
1. `docs/planet-layer/STATE.md`: what is running, the model as it stands (including the PARAMS-vs-.cmd warning),
   the scorecard against Earth, and the open problems in order. **This says what to do next.**
2. `docs/planet-layer/decisions.md`: skim it so you don't re-propose something already rejected.
3. `docs/project-map.md` (code, scripts, data, commands) and `docs/planet-parameter-audit.md` (which values are
   measured, assumed or fitted).
4. Journal entries (`docs/planet-layer/journal/`) only when STATE or decisions points to one. Don't read the journal
   front to back, and don't read `docs/planet-layer-plan.md` (the frozen old plan doc, ~360 KB).

## How to keep the docs (every pass)

- **Journal:** write each pass as a new file `docs/planet-layer/journal/NN_YYYY-MM-DD_pass-KK.md` (next: `59_…_pass-41`;
  non-pass entries get a short slug), with a `### ` heading as before. Add its row to `journal/README.md`. Never edit
  an old entry. Correct it in a later one.
- **STATE.md:** rewrite it so it is true after the pass: section 1 (running / to run), the model table, the scorecard
  (state the source sweep), and the open problems. Keep it under ~200 lines.
- **decisions.md:** add one row per adopted / rejected / kept setting, fix or rule, in the same pass.
- **This file:** change it only when a working rule changes.
- **Audit and project map:** update them when values, scripts or `.cmd` files change.
- **Project mirror** (claude.ai Project "Planetary Diffusion"): after a pass, upload `STATE.md` →
  `claude/planet-layer-state.md`, `decisions.md` → `claude/planet-layer-decisions.md`, this file →
  `claude/next-chat-prompt.md`, the audit → `claude/planet-parameter-audit.md` and the project map →
  `claude/project-map.md` if changed, plus the new journal file → `claude/planet-layer-journal/<file name>`.
  Use `project_write` with `local_path` (stage from the device, copy into the cloud workspace). Don't read the big
  Project docs just to compare. `claude/planet-layer-plan.md` there is the frozen old plan doc.
- **New chat:** tell the user when it is a good time to start one, once the docs and the Project copies are current.
  Do this when the chat has grown long or a checkpoint is reached (results read, docs updated, nothing half-built).
  Say it plainly, e.g. "This is a good point to start a new chat; the handoff is up to date."

## Working rules

**Direction and decisions**
- **Realism** (user, 23 Oct). Any system may be added, removed or changed if it moves toward realism. Comments may not
  reflect the goal or the code, so decide by the goal, not by a comment. Values come from Earth measurements or from
  mechanism, never from fitting the planet's output to a band. Replace values that only make a band pass.
- **Bring a decision to the user only when it is very important** (user, 26 Oct). That means one of these:
  - it cannot be undone cheaply;
  - it changes the project's goal or direction;
  - it needs something only the user has (data, hardware, money, or a long run on their machine that is not the
    routine sweep).

  Everything else Claude decides by the goal and these rules, and writes the reasoning in the journal: model design,
  which candidate to build next, test bands from Earth data, adopting by a rule fixed before the runs. Running the
  next `.cmd` sweep is routine: just say which one to run.
- Mechanism above ~100 km, learning below. Propose a new model only together with its test set.
- Fix the pass bands and the selection rule before running. Do a total re-evaluation when something looks off.
- Don't judge a law on a transient.
- **Sweeps decide on six seeds (1001–1006), never on one.**
- Always compare against `outputs/meso/dataset.npz`, and against ETOPO on the same grid (`scripts/earth_reference.py`).
- Avoid these:
  - learned planners above ~100 km;
  - artificial constraints (locks, frozen sign, projections, mean conservation);
  - scale hacks;
  - over-tuning the test bed;
  - training past the best point;
  - metric gaming.
- Judge terrain on numbers, or on pictures with enough colour levels (`elevation_shading.CMAP` N = 65,536; the old
  256 levels drew false terraces).

**Runs and identity**
- Don't run long generations or trainings yourself. Give simple PowerShell commands with real paths: no
  `<placeholders>`; use `--plates latest` / `--plates sweep`. Sweeps go in a root `.cmd` that also runs the tests and
  the reviews into `<sweep>/reviews/`.
- Runs on the user's machine are deterministic. The cloud workspace and the device shell follow their own
  trajectories (seed 1001 in the cloud: 26–28 plates at 0 Myr vs 20 on Windows). Use them for pre-checks only.
- Runs drift apart from the first steps. Measure an effect smaller than that drift paired, on the same states.
- Step-by-step identity checks need `params={'threads': 1}` (ducc0 threads differ at 1e-12). When editing the default
  path, check that it stays identical step by step (n = 48, ~70 steps).
- Sweep folders are named only by the parameters that differ (`MAX_NAME` 80, else `combo-NN`). Shared parameters go in
  `sweep.json`, per-combination ones in `<combo>/params.json`. Runs save `params_effective`. Reviews read older runs
  with `crust_review.LEGACY_DEFAULTS`.
- `planet_meso_check --landscape latest` prefers the newest theta calibration, so pass explicit landscape folders.
- Continental crust in the reviews is label-free: total (crust + sediment) ≥ 20 km on both sides (`crust_review` at
  46 km; `margin_width` on CRUST1.0's 1° cells via `to_crust1_cells`). The flag reading is printed beside it.
- Planet-layer tests (86 since the fortieth pass): `pytest -q tests\test_shelf.py tests\test_plates.py
  tests\test_sphere_graph.py tests\test_graph_landscape.py tests\test_coupling.py tests\test_boundary_lengths.py
  tests\test_erosion_law_check.py tests\test_crust_reviews.py`.
- The user's `.venv` has no scipy or scikit-image, so planet-layer code stays numpy / numba / ducc0 (+ matplotlib).

**The user's machine through the device shell** (device_bash; repo at `$HOME/mnt/Infinite_Diffusion`)
- Reviews and probes run there with `NUMBA_CACHE_DIR=$HOME/numba-cache PYTHONDONTWRITEBYTECODE=1`, and
  `PYTHONPATH=$PWD` for ad hoc scripts. Each call is ≤ 180 s, and background jobs are stopped. It has 2 cores and
  3.9 GB. Save review printouts beside the sweep (`<sweep>/reviews/*.txt`).
- Its Python packages can vanish between sessions. Reinstall with `pip install --user --break-system-packages numpy
  numba ducc0 pytest h5py`. Without numba the reviews still run on the repo's pure-Python fallback.
  `python3 tools/device_stubs/run.py <test names>` smoke-tests plate code with stubs. That is plumbing only, never
  for judging.
- Full n = 217 runs there use `tools/device_runs/chunk_run.py` + `simpickle.py`. They are pickled between 180 s
  calls, two in parallel take ~12 min per run, and they are for pre-checks only.
- `test_coupling` must run from a copy (`$HOME/work/tc/tests`): the shell cannot delete its old `outputs/test-tmp`.
  The shell cannot delete files at all without the user's permission.
- Edit files in place with short Python read-modify-write scripts that assert each old string matches once. Check the
  md5 before and after.
- Long shell commands fail (`spawn E2BIG`). Write big files in the cloud workspace and copy them with
  `device_commit_files` from `/mnt/user-data/outputs/<fresh folder>/<new name>`. Use a new file name for every sync
  (a re-commit from the same staged path can write old content). Check the device md5 before overwriting and verify
  it after.
- `pkill -f` patterns can match your own command and kill your shell. Kill by PID from
  `ps aux | grep "simulate_[p]lates"`.

**Cloud workspace**
- It can do real runs: `pip install --break-system-packages numpy numba ducc0 pytest`, then the code staged as
  tarballs. Use planet_layer, planet_diffusion, planet_training, toy_meso, scripts, tests, tools, data/crust1 and
  data/tectonicplates-master. There is no ETOPO there, so `elevation_check` does not run. Two 250 Myr n = 217 runs in
  parallel with `--param threads=1` take ~25–60 min (Bash `run_in_background`).
- Neither shell reaches most data servers. The user downloads data in a browser.
