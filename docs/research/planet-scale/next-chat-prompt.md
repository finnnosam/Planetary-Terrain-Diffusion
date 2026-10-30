I'm building a planet terrain generator in `E:\projects\Infinite_Diffusion` (Windows, PowerShell, `.venv`, RTX 3070 with 8 GB VRAM). Please read these first, in this order. They are in the folder and also in the attached Project:
1. `docs/research/planet-scale/README.md` (Project copy: `claude/planet-scale-research.md`): the research survey for this step.
2. `docs/tectonic-prototype.md` (Project copy: `claude/tectonic-prototype.md`): the plan of record and every result so far.
3. `claude/scale-review.md` and `claude/training-data-plan.md` (Project): why scale matters and how each model learns Earth at one spacing.
This was originally a 'simpler' project I uploaded to Github, but since starting training, is now the version that won't be uploaded. I will make a separate one that only contains necessary data at a later date for sharing.

## Goal of this chat
Design, and then start building, the **planet-scale layer**: what decides where continents, plate boundaries, mountain belts, old worn ranges, cratons, rifts and basins sit on a whole planet. It must feed a landscape-evolution stage on the cube sphere, which then feeds the existing meso model at 23.04 km. Start by turning the research survey into a concrete design. Compare the options there honestly:
- Cortial-style plate kinematics, plus our own landscape model ported to the cube sphere
- goSPL
- fastscapelib's HEALPix grid
- anything better you find

Then give me the design to agree on before building. After agreement, build it in small, testable pieces.

## Current direction (agreed)
- **Mechanism above ~100 km, learning below.** Tectonic uplift plus landscape evolution (rivers, sediment, flexure, drainage basins as part of the process) makes the large structure in metres and years. Learned models add detail only below 23 km.
- **The chain that works today:**
  1. `planet_training/landscape_evolution.py`: regional, 11.52 km, numba. Processes: priority flood, D8 routing, Yuan 2019 implicit erosion and deposition, deltas and shelves, hillslope and marine diffusion, FFT flexure with Te 30 km.
  2. 2×2 area mean down to 23.04 km.
  3. The meso, a diffusion model from 23.04 km to 2.88 km, 128×128 patches. Default checkpoint: `outputs/meso/detail-ft/step10000-passed.pt`. It was fine-tuned from `outputs/meso/run2-lowlr/latest.pt` with softened inputs, plus an optional per-cell **detail map** input and a known/unknown mask (`toy_meso/detail.py`, `toy_meso/finetune_detail.py`). With the true detail map it is close to Earth everywhere; without one it guesses slightly rough. **So the planet layer should eventually supply a detail map**, for example from rock age, erodibility and recent uplift.
- `scripts/simulate_tectonic_region.py` and `planet_training/tectonic_templates.py` are **only a test bed** for the processes, not the generator. Don't spend effort tuning them.
- The cube-sphere machinery already exists in `planet_diffusion/cube.py`: equi-angular charts, halos, area weights. Reuse it.
- Two clocks: plates need ~100–250 Myr of history; erosion at 11.5 km needs 50 kyr steps. A global 11.52 km cube is ~3.8 M cells, 59× the test window, which takes ~30 s per 30 Myr. A likely shape is coarse plates over a long time, then a fine landscape model for the last ~20–40 Myr. Decide this with numbers.

## Misdirections already made: don't repeat them
- **Learned planners and layout models above ~100 km** (structure plans, continent models, terrain-plan/oracle/joint-structure experiments, the PaleoDEM layout model "L") produced smooth ramps and gradients. Earth offers too few examples at that scale. They are superseded.
- **Artificial constraints:** locks, frozen land/ocean sign, sea-level projection, mean conservation, "feasible targets", degraded or translated training guides. The user does not want them ("freezing anything sounds incorrect"). Sea level is only the 0 m datum.
- **Scale hacks:** shrinking the planet, coarse pooling, cascades. Each draws landforms at the wrong size. A model must run at the spacing it was trained at.
- **The "adapted" meso,** fine-tuned to smoothed guides, lost plain texture. Use the current default checkpoint.
- **Comparing against the wrong reference.** I once compared simulated 23 km fields with a different Earth dataset (`bridge-adapted`) instead of the meso's own validation inputs, and missed the real gap. Always compare against what the model was trained on (`outputs/meso/dataset.npz`).
- **The smooth-plains gap** turned out to be missing cell-to-cell curvature at 23 km: a model with 11.5 km cells can't make 20–50 km valleys. Weakening diffusion or deposition didn't help. A 5.76 km grid closes only ~25% of the gap. The fix was the meso fine-tune, not more tuning of the simulation.
- **Over-tuning the tester,** and proposing refinements (water balance) before the planet structure exists. Prioritise what moves the actual generator forward.
- **Training past the best point.** The meso fine-tune overfit after ~10–20k steps (held-out loss rose 4%). Fine-tunes now keep `best.pt`. Judge models by held-out tests decided in advance, not by training loss.

## Working rules
- **Don't run long generations or trainings yourself.** Give me a simple PowerShell command and I'll run it on my GPU. Short CPU tests in your workspace are fine.
- **On my 8 GB GPU,** training needs `--data-on-cpu` and usually `--batch 16`. Data held on the GPU spills into Windows shared memory and crawls.
- **Do a total re-evaluation when something looks off.** Something already happening isn't automatically good.
- **Propose a new model only if it would actually be better,** and set the test that decides it before training.
- **Later, not now:** a climate layer, including a precipitation–evaporation balance so closed basins stay closed and fills aren't dead-flat; and a multi-input meso conditioned on rivers, climate and similar.
- **If you edit files on my computer through the bridge:** stage each sync in a fresh folder and verify md5 on my machine afterwards. Re-committing through the same staged path once silently failed.
- **Validation targets** are listed in the research README: Earth hypsometry, plate size distribution, Hasterok 2022 province proportions, the 368 km land-relief distribution (Earth p10/p25/p50/p75 = 133/237/456/872 m), 40 km arrangement metrics, 23 km curvature per relief class, and `scripts/diagnose_meso_plains.py` end to end.
- **Update `docs/tectonic-prototype.md`,** or a new plan doc for the planet layer, as decisions are made. Mirror it to the Project.
