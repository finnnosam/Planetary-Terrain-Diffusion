"""Compare regional meso outputs on exactly the original global TIFF pixel grid."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from toy_meso.planet_tiled import main as generate, Grid, blend, TileCache
from toy_meso.sphere import GlobalRaster, decode
from toy_meso.evaluate import radial_psd, hillshade
from planet_diffusion import cube


def main():
    torch.set_num_threads(4)
    before = Path('human_input/MESOFromIntermediate/planet.tif')
    after_folder = Path('outputs/meso/adapted-planet42')
    after_folder.mkdir(parents=True, exist_ok=True)
    old_guide = GlobalRaster.from_geotiff('outputs/intermediate/evaluation-planet42/intermediate/planet.tif')
    new_guide = GlobalRaster.from_geotiff('outputs/intermediate/adapted-planet42/planet.tif')
    places = [('Southern highland', -35.564516, -51.947005),
              ('Northern highland', 37.638249, -2.799539), ('Lowland', 10., -75.)]
    cases, records = [], []
    with rasterio.open(before) as source:
        for index, (name, lat, lon) in enumerate(places):
            row, column = source.index(lon, lat)
            row -= 128; column -= 128
            width = height = 256
            west = source.bounds.left+column*source.transform.a
            north = source.bounds.top+row*source.transform.e
            east = west+width*source.transform.a
            south = north+height*source.transform.e
            dest = after_folder/f'region-{index+1}.tif'
            if not dest.exists():
                generate(['--run', 'outputs/meso/run2-lowlr', '--checkpoint', 'latest.pt',
                          '--dem', 'outputs/intermediate/adapted-planet42/planet.tif',
                          '--out', str(after_folder), '--name', dest.name,
                          '--bounds', *map(str, (west, south, east, north)),
                          '--width', str(width), '--height', str(height),
                          '--seed', '1135849674', '--steps', '32', '--batch', '16', '--device', 'cuda'])
            original = source.read(1, window=Window(column, row, width, height))
            with rasterio.open(dest) as ds:
                modified = ds.read(1)
                assert np.allclose(tuple(ds.transform), tuple(source.window_transform(Window(column, row, width, height))), atol=1e-10)
                assert np.isfinite(modified).all()
            latitude = north+(np.arange(height)+.5)*source.transform.e
            longitude = west+(np.arange(width)+.5)*source.transform.a
            guide0 = old_guide.sample(latitude[:, None], longitude[None, :])
            guide1 = new_guide.sample(latitude[:, None], longitude[None, :])
            h = np.pi*6371000/source.height
            wl, p0 = radial_psd(original[None], h)
            _, p1 = radial_psd(modified[None], h)
            band = (wl >= 6000)&(wl < 23000)
            records.append({'name': name, 'bounds': [west, south, east, north],
                'after': str(dest), 'shape': [height, width],
                'nominal_6_23km_power_new_vs_old': float(p1[band].sum()/p0[band].sum()),
                'output_minus_input_guide_rms_m': {'old': float(np.sqrt(np.mean((original-guide0)**2))),
                                                 'new': float(np.sqrt(np.mean((modified-guide1)**2)))}})
            cases.append((name, guide0, guide1, original, modified, h))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, len(cases), figsize=(12, 14), squeeze=False)
    for c, (name, g0, g1, z0, z1, h) in enumerate(cases):
        lo, hi = np.percentile(np.r_[g0.ravel(), g1.ravel(), z0.ravel(), z1.ravel()], [1, 99])
        for r, (label, z) in enumerate([('Original intermediate', g0), ('Adapted intermediate', g1),
                                       ('Original meso result', z0), ('Meso with adapted input', z1)]):
            rgb = plt.cm.terrain(np.clip((z-lo)/max(hi-lo, 1), 0, 1))[..., :3]
            rgb *= .3+.7*hillshade(z, h)[..., None]
            axs[r, c].imshow(rgb, interpolation='nearest')
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if c == 0: axs[r, c].set_ylabel(label)
            if r == 0: axs[r, c].set_title(name)
    fig.suptitle('Actual planet pipeline: same continent layout, meso checkpoint and meso noise seed\n'
                 'Same output pixel grid and display per region; ~740 km north–south close-ups')
    fig.tight_layout(); fig.savefig(after_folder/'comparison.png', dpi=140); plt.close(fig)

    # Check large-scale elevation fidelity directly on each native bridge cube.
    consistency = {}
    for label, folder in [('original', Path('outputs/intermediate/evaluation-planet42/intermediate')),
                          ('adapted', Path('outputs/intermediate/adapted-planet42'))]:
        settings = json.loads((folder/'settings.json').read_text())
        grid = Grid(settings['fine_intervals'], settings['tile'], settings['stride'], settings['factor'])
        keys = grid.all_tiles()
        acc, weights = blend(keys, grid, TileCache(folder/'tiles'))
        values = decode((acc/weights)[0])[:, :grid.N, :grid.N]
        area = cube.area_weight(np.arange(grid.N)[:, None], np.arange(grid.N)[None, :], grid.N)
        f, n = grid.factor, grid.N//grid.factor
        average = (values*area).reshape(6, n, f, n, f).sum(axis=(2, 4))
        average /= np.broadcast_to(area, values.shape).reshape(6, n, f, n, f).sum(axis=(2, 4))
        coarse = np.load(folder/'coarse.npy')[0, :, grid.coarse_halo:grid.coarse_halo+n,
                                               grid.coarse_halo:grid.coarse_halo+n]
        error = average-coarse
        consistency[label] = {'coarse_mean_error_rms_m': float(np.sqrt(np.mean(error**2))),
                              'coarse_mean_error_p95_abs_m': float(np.percentile(abs(error), 95))}
    with rasterio.open('outputs/intermediate/adapted-planet42/planet.tif') as ds:
        z = ds.read(1); assert np.isfinite(z).all()
        weights = np.cos(np.radians(90-(np.arange(ds.height)+.5)*180/ds.height))[:, None]
        globe = {'shape': list(z.shape), 'land_fraction': float(((z>0)*weights).sum()/(weights.sum()*ds.width)),
                 'min_m': float(z.min()), 'max_m': float(z.max())}
    report = {'regions': records, 'bridge_coarse_fidelity': consistency, 'adapted_globe': globe,
              'note': 'Three selected regions, one planet, one seed. Spectral bands use nominal equirectangular north–south spacing; east–west physical spacing varies with latitude. Higher power alone does not establish terrain realism.'}
    (after_folder/'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
