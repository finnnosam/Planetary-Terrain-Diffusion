"""Earth's elevation statistics on the planet layer's own grid: ETOPO 2022 (30", surface) area-averaged onto the nodes
of planet_layer.sphere_graph.CubeSphere(n), so planet and Earth are compared at the same spacing.

  python -m scripts.earth_reference                 # n = 217 (46 km), writes outputs/planet-reference/earth-n217.{npz,json}

Reported for the whole Earth and without the ice sheets (Antarctica south of 60 S, Greenland), since the planet has
no ice: land fraction, land / ocean elevation percentiles, mean land elevation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import fields
from planet_layer.sphere_graph import CubeSphere

ROOT = Path(__file__).resolve().parents[1]
ETOPO = ROOT/'data'/'etopo_2022_30s_surface.tif'


def node_means(path, g, rows_per_chunk=200):
    """Area-weighted (cos latitude) mean of every raster pixel, assigned to its nearest cube node."""
    import rasterio
    s = np.zeros(g.size); w = np.zeros(g.size)
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        lon = ds.transform.c+ds.transform.a*(np.arange(W)+.5)
        for r0 in range(0, H, rows_per_chunk):
            r1 = min(H, r0+rows_per_chunk)
            a = ds.read(1, window=((r0, r1), (0, W))).astype(np.float64)
            if ds.nodata is not None:
                bad = a == ds.nodata
            else:
                bad = np.zeros(a.shape, bool)
            lat = ds.transform.f+ds.transform.e*(np.arange(r0, r1)+.5)
            la, lo = np.radians(np.meshgrid(lat, lon, indexing='ij'))
            p = np.stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)), -1).reshape(-1, 3)
            nodes = fields.nearest_nodes(g, p)
            wt = np.cos(la).ravel()*~bad.ravel()
            s += np.bincount(nodes, weights=a.ravel()*wt, minlength=g.size)
            w += np.bincount(nodes, weights=wt, minlength=g.size)
            print(f'  rows {r1}/{H}', end='\r', flush=True)
    print()
    return s/np.maximum(w, 1e-12)


def stats(z, area, mask):
    land = mask & (z > 0); ocean = mask & (z <= 0)
    q = (10, 25, 50, 75, 90, 99)
    return {'land_fraction': float(area[land].sum()/area[mask].sum()),
            'land_mean_m': float(np.average(z[land], weights=area[land])),
            'land_percentiles_m': {str(k): float(np.percentile(z[land], k)) for k in q},
            'ocean_percentiles_m': {str(k): float(np.percentile(z[ocean], k)) for k in q}}


def ice_mask(g):
    lat = np.degrees(np.arcsin(g.xyz[:, 2])); lon = np.degrees(np.arctan2(g.xyz[:, 1], g.xyz[:, 0]))
    greenland = (lat > 59) & (lon > -75) & (lon < -10) & ~((lat < 66) & (lon > -40))   # rough box, excludes Iceland
    return (lat < -60) | greenland


def load(n=217):
    """Cached Earth reference for cube n (computed on first use)."""
    out = ROOT/'outputs'/'planet-reference'/f'earth-n{n}.json'
    if not out.exists():
        main(['--n', str(n)])
    return json.loads(out.read_text())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=217)
    p.add_argument('--etopo', default=str(ETOPO))
    a = p.parse_args(argv)
    g = CubeSphere(a.n)
    print(f'ETOPO -> cube n={a.n} ({g.size:,} nodes)', flush=True)
    z = node_means(a.etopo, g)
    out = ROOT/'outputs'/'planet-reference'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f'earth-n{a.n}.npz', elevation_m=z.astype(np.float32), n=a.n)
    ice = ice_mask(g)
    res = {'source': str(a.etopo), 'n': a.n, 'all': stats(z, g.area, np.ones(g.size, bool)),
           'no_ice_sheets': stats(z, g.area, ~ice)}
    (out/f'earth-n{a.n}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
