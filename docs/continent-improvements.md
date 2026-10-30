# Continent continuation and controlled globe, 11 October 2026

The original continent model was continued for 12,000 additional updates with
CUDA batch 16, learning rate 0.00003 and a fresh optimizer initialized from the
original EMA at update 30,000. Architecture, data splits and normalization were
preserved. The selected checkpoint is frozen at:

`outputs/continents/selected-continued-42000/best.pt`

SHA256: `8c4db20e6ecd2664c8307cb1efdeb2537acffb1b04029d9994e2e7e2d8485661`

`outputs/continents/selection.json` records the selection. The original `run1`
checkpoint is retained. `Train Continents Continued.cmd` can resume the separate
continuation experiment; it does not replace this frozen selection.

## Evaluation and practical limits

The full 18-age validation loss decreased from 0.16759715 to 0.16443329 (1.89%).
Sixteen matching seeds, each sampled for 64 steps, were compared at native
spherical resolution using area weighting and land components joined across
cube edges. All 67 training and 18 validation maps were reference populations;
the five temporal test maps were left untouched.

| Median over the same 16 seeds | Original | Continued |
|---|---:|---:|
| Natural land coverage | 16.13% | 15.80% |
| Land masses covering at least 1% of planet area | 4 | 3 |
| Fraction of land in the largest land mass | 40.35% | 42.69% |
| Land elevation 95th percentile | 1,105 m | 1,080 m |
| Approximate best training-map land-mask IoU | 0.316 | 0.355 |

Continuation gives a modest improvement in loss and land concentration, rather
than taller mountain ranges. Generated continents remain more fragmented than
the PaleoDEM references (median two large land masses, with about 77–79% of land
in the largest). Natural coverage remains low, so explicit coverage control is
useful. The similarity screen checks all training layouts with rotations and
reflections, using 2,048 equal-area probes and local orientation refinement.
Maximum generated IoU is about 0.49, but this approximate screen cannot certify
the absence of copying.

Full metrics, matching-seed previews, distributions and spherical spectra are in
`outputs/continents/run2-lowlr/evaluation/`. These tests evaluate broad layout;
the continent model's ~161 km resolution cannot represent individual valleys.

## Controls and export correction

Double-click `Launch Continents.cmd` for the new continent interface. It follows
the frozen selection and offers seed, land percentage and land-relief controls.
Blank coverage and relief 1 preserve the model's natural result. Coverage shifts
sea level using native spherical area weights; relief multiplies positive land
heights after that shift. These are generation-time elevation edits, not learned
conditioning or new tectonic structure.

The continent GeoTIFF exporter now samples actual area-raster cell centres.
Previously, it passed the node-preview array (including both poles and an extra
endpoint row) into a smaller area raster, causing implicit resizing and a
longitude half-pixel shift. Native cubes and correctly aligned TIFFs are now
saved together.

100 targeted checks passed across warm-start/resume behavior, layout geometry,
controls, GUI launch settings and existing meso conditioning/tiled generation.
Compilation passed. The final meso artifact was inspected after completion;
see `docs/structured-continent-overhaul.md` for the results and the user's
decision to replace post-generation sea-level controls with coastline-first
generation.

## Chosen planet and current generation

The user chose **30% land with 1.25× relief**, using seed 42 and 64 continent
sampling steps. The comparison is
`outputs/continents/continued-examples/options.png`.

Chosen layout:
`outputs/continents/continued-examples/land30-relief125/layout.tif`

The native layout achieves 30.00295% land, with a -673.07 m sea-level offset,
four large land masses and 1,730.48 m land-elevation 95th percentile. Its exported
pixel-centred TIFF has 29.7103% area-weighted land because of interpolation.

The full intermediate pass is **complete**, using frozen adapted update 1,750,
seed 42, 32 sampling steps and CUDA batch 16:

`outputs/intermediate/continued-land30-relief125/planet.tif`

It contains 1,736 × 868 finite metre elevations spanning -5,609 to 4,626 m;
area-weighted land coverage is 29.8922%.

The full meso pass is **complete**, using frozen
adapted update 1,000, seed 42, 32 sampling steps and CUDA batch 16. It has 8,664
tiles and ran at about 2.83 tiles/s, roughly 51 minutes of denoising
plus final blending/export. Final output is saved at:

`outputs/meso/continued-land30-relief125/planet.tif`

This final file is published only after export completes. Completed tiles are
cached, so an interrupted run can resume with the same settings. The process was
left running at the user's request, then verified after completion. Its
13,888 x 6,944 elevations are finite, with 29.813% area-weighted land coverage.
No generation or training was interrupted.

To resume the same meso run if needed:

```powershell
.venv\Scripts\python.exe -u -m toy_meso.planet_tiled --run outputs/meso/selected-adapted-1000 --checkpoint best.pt --dem outputs/intermediate/continued-land30-relief125/planet.tif --out outputs/meso/continued-land30-relief125 --seed 42 --steps 32 --batch 16 --device cuda
```

The completed output was inspected using:

```powershell
.venv\Scripts\python.exe -m scripts.verify_continued_planet
```

This writes `verification.json`, `stage-closeups.png`, `globe-overview.png` and
three native-resolution close-up TIFFs alongside the final globe. It selects
separated highland/upland/lowland examples from the intermediate guide before
inspecting meso detail, compares stages on the same final pixel grid, verifies
finite elevations and measures global area-weighted land coverage. Regional
6–23 km power uses nominal north–south spacing; east–west spacing varies with
latitude. Three examples of one seed and greater power alone do not establish
terrain realism or connected drainage. Adapted models still underproduce fine
relief on strongly smoothed held-out inputs; narrow rivers are below 2.88 km
resolution. Final land percentage can differ slightly from the layout target.
