# Upstream inspection and smallest viable approach

Historical v1/v2 design. The upstream inspection remains relevant, but the
latitude/longitude generation grid below was superseded after polar stretching
was observed. See [the v3 cubed-sphere design](cubed-sphere-v3.md) for the current
implementation, including coarse and detail generation.

Inspected upstream commit `e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230`:

- [`world_pipeline.py`](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/inference/world_pipeline.py):
  planar tile-seeded noise; coarse 64px denoising tiles; latent 64px tiles
  conditioned by 4×4 coarse crops; 32× coarse-to-latent and 8× latent-to-height
  scale factors; decoder normally 512px. Its periodic custom conditioning
  independently takes row/column modulo: a torus, not a sphere. Merely wrapping
  that input would leave diffusion noise, intermediate states and decoder
  contexts inconsistent at the boundaries.
- [`edm_unet.py`](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/models/edm_unet.py):
  the reusable pretrained model class, with separate coarse, base and decoder
  weights. Importing it directly avoids the full planar pipeline.
- [`laplacian_encoder.py`](https://github.com/xandergos/terrain-diffusion/blob/e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230/terrain_diffusion/data/laplacian_encoder.py):
  residual plus resized low-frequency signed-square-root elevation, followed by
  `sign(z)*z²` for metres. The planar extrapolation/filtering is unsuitable at
  sphere boundaries. Version 2 substitutes spherical interpolation, antialiased
  reduction and an 11-tap sigma=5 Gaussian with spherical halos. Version 1
  incorrectly omitted this low-frequency denoise pass, producing latent-grid
  knots visible as rows and columns of blobs.

## Decision

Reuse the three networks and upstream EDM solver unchanged. Replace the
unbounded planar execution graph with a finite shared spherical state. This is
smaller than modifying every InfiniteTensor cache/dependency address or training
a spherical network. A cubed-sphere sampler would provide better polar metric
behaviour, but introduces chart rotations, latent transport and multiscale
face-edge/corner resampling; that is beyond this minimal reference implementation.

## Topology and generation

An internal field has shape `(channels, H+1, 2H)`. Rows are latitude nodes,
including both poles. Longitudes are periodic with no duplicated endpoint.
All channels are treated as scalar latent fields. Canonicalize a row `y` by
reducing modulo `2H`; rows beyond `H` reflect to `2H-y` and shift longitude by
`H` (180°). Both pole rows represent one value each.

The initial geographic sketch is a seed-driven smooth function of Cartesian
unit-sphere coordinates, not planar noise. Initial noise and every injected
noise field are generated once on the canonical global grid. Every patch reads
that same field through spherical addressing. This intentionally sacrifices
upstream's constant-time random access for a simple auditable reference.

At each coarse diffusion timestep:

1. Extract overlapping 64×64 patches across longitude and both poles.
2. Evaluate the pretrained denoiser using equally addressed conditioning.
3. Fold and accumulate weighted predictions into a single spherical prediction.
4. Advance one global EDM solver (including its shared multistep history).
5. Apply the polar constraint before the next model evaluation.

The latent consistency model uses the same prediction consensus before each
of its two updates. The decoder uses overlapping 512px model contexts and
consensus before its single update. Model-window weights act on predictions
during generation, never independently completed heightmaps. Every stage,
including conditions, injected noise and decoder residual, has identified poles.

Only the first ring beside each pole has its denoised modes limited to 0 and 1;
all other interior samples are unchanged by the polar projection. Each pole
row is identified to a single scalar value. Noise innovations are unit Gaussian
at every node, with one shared Gaussian draw at each pole, and are never
low-pass filtered. Version 1's full-row Fourier taper changed the model's noise
variance and suppressed learned detail at every latitude; that was a defect.
The continuous piecewise-bilinear spherical interpolant has a unique limit at
each pole and across the date line. It guarantees C0 continuity, not a globally
smooth derivative or exact rotation equivariance. The final signed-square
conversion is pointwise and preserves continuity; no final seam repair exists.

The decoder's residual and low-frequency latent channel are reconstructed using
the upstream sequence: provisional decode, antialiased reduction, low-frequency
Gaussian denoise, and addition of the unchanged learned residual. This is a
model reconstruction operation on the shared field, not a finished-image seam
blend. A numerical regression measures gradient strength at each phase of the
8-pixel latent grid and catches the periodic interpolation knots seen in v1.

## Tiles and repeatability

Generate the global native detail state once. Global overviews and tiles are
evaluations of exactly that state at specified global pixel centres. Two tile
requests at the same grid resolution use identical coordinates even when one
crosses 180°. Seed, domain size, checkpoint, steps and software/device together
identify a run. The seed alone does not define resolution-independent terrain.
The saved NPY state provides exact repeatable exports without running models.

## Limitations

- This is a topology-correct experimental sampler using planar pretrained
  networks, not a pretrained spherical diffusion model. Polar coherence is a
  shared-state constraint; natural polar geology requires empirical evaluation
  and likely retraining or a cubed-sphere/local tangent-chart design.
- Equirectangular metric distortion remains, despite corrected unit-variance
  noise. Latent channels are transported as
  scalars even though the learned representation need not be rotation invariant.
- The smallest coarse grids repeat many times inside a 64px context. They are
  useful for smoke tests but severely outside the training distribution.
- Memory and work scale with global native area; no sparse tile generation,
  resumable stage cache, global hydrology, sea-level balancing, or geoid model.
- The output elevation amplitude uses the model's metre calibration, while
  horizontal scale comes from radius/grid size. Rescaling the planet does not
  make the pretrained geomorphology physically scale invariant.
- Mathematical boundary tests alone cannot detect all visible seams or
  anisotropic terrain. Learned-output statistics must be assessed separately.

Upstream code remains in a separate pinned checkout under its own MIT license;
its scheduler also contains Apache-2.0 notices. No upstream source is copied
into this package.
