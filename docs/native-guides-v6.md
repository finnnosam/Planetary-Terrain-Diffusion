# Native guide refinement (0.6)

Version 0.5 over-corrected the weak PNG import. Its excellent land-mask agreement
did not establish good terrain: it clamped learned low-frequency elevations to
the drawing and exempted the drawing from reconstruction smoothing. This could
retain painted plateaus and suppress terrain variation. Its ocean fallback also
switched abruptly from the -2000 m prior to a much shallower coarse prediction
at a fixed threshold. That introduced a mechanism for a deep coastal ring,
followed by shallower water, followed by deep ocean. These additions are removed.

An audit of the v0.5 Earth run found 527 horizontal threshold crossings with
both endpoints in essentially pure ocean. The substituted ocean target jumped
by 1,943.6–1,975.0 m (median 1,967.8 m) at those crossings, before latent
constraints/reconstruction. This is evidence of the introduced artifact's
mechanism, not a final-raster slope measurement. The audit is saved in
`outputs/native-guide-probes/coastal-fallback-audit.json`.

## What upstream actually supports

In the pinned checkout, `terrain_diffusion/inference/tiff_export.py` imports
height and optional climate TIFFs with `set_custom_conditioning_import` and
`set_cond_snr`. The recommendation is `[0.2, 0.2, 1.0, 0.2, 1.0]` for elevation,
temperature, temperature variation, precipitation and precipitation variation.
Despite the SNR name, these values parameterize the amount of conditioning noise;
smaller values preserve the guide more closely. The elevation channel is first
converted to signed square-root metres. The coarse model refines it, the latent
model generates from that result, and the decoder/reconstruction produce heights.
No painted low-frequency map is forced back into the latent or final field.

The cube sampler now uses that conditioning/noise-label convention, while keeping
its spherical noise, shared nodes and cross-face model contexts. It cannot call
upstream's planar WorldPipeline directly without losing spherical topology.
There is no additional draft constraint after coarse inference or inside latent
updates or reconstruction. The models generate both relief and coastal profiles.

Black supplies a negative coarse elevation prior (-2000 m in automatic mode).
That is a PNG-import convention, not a native learned ocean-mask capability.
Upstream expects actual elevations, including underwater elevations; it does
not infer a hidden depth map from the meaning of black. The negative prior is
refined by the model and is no longer substituted into the detail field.

## Resolution is part of guide fidelity

Upstream expands one guide cell into 256 output pixels on each axis. A 4k global
cube export has only 98 unique coarse nodes, 8k has 386, and 16k has 1538.
An 8192×4096 input is therefore a source sketch to downsample; it is not 8192×4096
independent conditioning cells. Feeding all its pixels as upstream guide cells
would instead require a 2,097,152×1,048,576 planar output.

Same-seed, real pretrained coarse-only probes on the Earth draft, with white
6250 m and refinement 0.2, gave these area-weighted land/ocean agreements:

| Global native output size | Coarse-only agreement |
|---|---:|
| 4096×2048 | 79.94% |
| 8192×4096 | 89.55% |
| 16384×8192 | 93.93% |

These probes are not finished heightmaps or terrain-quality scores. Their JSON
and coarse states are in `outputs/native-guide-probes/`. They show why a larger
generation grid is needed instead of overwriting learned detail to recover the
input outline. Full 8k generation is about four times the work of 4k; 16k costs
about sixteen times as much in large-grid limits. CPU runs can take tens of
minutes or longer. The launcher now defaults to 8k and also offers 16k.

`--draft-refinement` (default 0.2, allowed 0.01..4) exposes the native elevation
conditioning-noise control. Smaller values favor the rough guide; larger ones
allow more reinterpretation. Neither exact coastlines nor 98% agreement is a
target. Fine terrain and recognizably related large-scale geography are the goal.

The same-seed 8k run is in `outputs/blobby-earth-native-guide/`, using
`--coarse-height 16 --draft-refinement .2 --draft-ocean-depth 0
--draft-white-metres 6250 --seed 2644365341129234824` and the original PNG.
Previous outputs are retained. Restart the launcher and use a new run folder;
old model checkpoints intentionally reject the changed algorithm identity.

## Completed 8k result

`outputs/blobby-earth-native-guide/planet-native.tif` is the finished 8192×4096
GeoTIFF; `planet-overview-4k.tif` is a 4096×2048 export of the same saved field.
The native raster contains finite elevations from -4162.125 to 6770.482 m.
The original PNG hash and seed were verified. Area-weighted land/ocean agreement
is 84.44%, with 80.53% of painted land remaining land and 86.00% of painted
ocean below sea level. These measure draft fidelity, not natural terrain quality.
The freer native conditioning changes coastlines substantially compared with
the old constrained result; exact outlines are not guaranteed.

Date-line identification and pole longitude spread are exactly zero in the
spherical field. The date-line raster gradient is 0.47 times the ordinary
longitude gradient, and an exported date-line tile matches the global raster
bit for bit. No latent-grid periodicity warning was raised. Cube shared-node
errors are zero; transverse edge gradient ratios range from 1.07 to 2.26, so
these checks do not establish derivative continuity across every cube edge.
No additional cube-edge correction was introduced for this guide fix.

The 37-test suite passed, including seeded repeatability and prevention of
downstream guide overwrites. The full 8k model run was performed once; its
saved-state digest and repeatable tile export were verified, not a second
complete 8k generation. Full measurements are in
`outputs/blobby-earth-native-guide/draft-layout-verification.json`.
