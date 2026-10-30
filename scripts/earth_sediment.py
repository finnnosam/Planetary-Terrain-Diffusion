"""GlobSed (Straume et al. 2019, G3 20, 1756-1772; NOAA NCEI) on the planet layer's grid: total sediment thickness of
the oceans and marginal seas, area-averaged onto planet_layer.sphere_graph.CubeSphere(n) nodes.

  python -m scripts.earth_sediment              # n = 217, writes outputs/planet-reference/globsed-n217.npz

Source file: data/GlobSed/V3/GlobSed_package3/ArcGIS_ascii/globsed-v3.asc (5 arc-minute grid, metres, cell centres
from -180 / -90 in steps of 1/12 degree; missing values written as '1.#QNAN' or -9999). The package's netCDF
(GlobSed-v3.nc) is HDF5 and needs h5py or netCDF4, which the project's .venv does not have; read with h5py on 26 Oct it
equals the ASCII grid within 0.006 m and equals GlobSed_v2.nc except that v2's few negative values (-69 m minimum) are
0 in v3. Area-weighted mean over the valid area (71.2% of the surface): 926.8 m. GlobSed thickness is seismic
(two-way time converted with a velocity-depth law): bulk, porous sediment, not solid volume.
"""
import argparse
from pathlib import Path

import numpy as np

from planet_layer import fields
from planet_layer.sphere_graph import CubeSphere

ROOT = Path(__file__).resolve().parents[1]
GLOBSED = ROOT/'data'/'GlobSed'/'V3'/'GlobSed_package3'/'ArcGIS_ascii'/'globsed-v3.asc'
OUT = ROOT/'outputs'/'planet-reference'


def read_asc(path=GLOBSED):
    """(grid north row first, lat of rows, lon of columns) from an ArcGIS ASCII grid; missing values as NaN."""
    raw = Path(path).read_bytes()
    lines = raw.split(b'\n', 6)
    head = {l.split()[0].decode().lower(): float(l.split()[1]) for l in lines[:6]}
    body = lines[6].replace(b'-1.#QNAN', b'nan').replace(b'1.#QNAN', b'nan').replace(b'-1.#IND', b'nan')
    nr, nc = int(head['nrows']), int(head['ncols'])
    a = np.array(body.split(), dtype=np.float32)
    assert a.size == nr*nc, (a.size, nr, nc)
    a = a.reshape(nr, nc)
    a[a == head.get('nodata_value', -9999.)] = np.nan
    cs = head['cellsize']
    lat = head['yllcorner']+cs*(nr-1-np.arange(nr)+.5)
    lon = head['xllcorner']+cs*(np.arange(nc)+.5)
    return a, lat, lon


def node_means(g, a, lat, lon, rows_per_chunk=200):
    """Mean of the valid pixels assigned to each node (cos latitude weights) and the valid share of the node's pixels."""
    s = np.zeros(g.size); w = np.zeros(g.size); wa = np.zeros(g.size)
    for r0 in range(0, a.shape[0], rows_per_chunk):
        r1 = min(a.shape[0], r0+rows_per_chunk)
        la, lo = np.radians(np.meshgrid(lat[r0:r1], lon, indexing='ij'))
        p = np.stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)), -1).reshape(-1, 3)
        nodes = fields.nearest_nodes(g, p)
        v = a[r0:r1].ravel().astype(np.float64)
        ok = np.isfinite(v)
        cw = np.cos(la).ravel()
        s += np.bincount(nodes, weights=np.where(ok, v, 0.)*cw, minlength=g.size)
        w += np.bincount(nodes, weights=cw*ok, minlength=g.size)
        wa += np.bincount(nodes, weights=cw, minlength=g.size)
    return s/np.maximum(w, 1e-12), w/np.maximum(wa, 1e-12)


def load(n=217):
    """Cached GlobSed on cube n: (sediment_m, valid share), computed on first use."""
    f = OUT/f'globsed-n{n}.npz'
    if not f.exists():
        main(['--n', str(n)])
    z = np.load(f)
    return z['sediment_m'].astype(np.float64), z['valid_share'].astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=217)
    ap.add_argument('--asc', default=str(GLOBSED))
    a = ap.parse_args(argv)
    g = CubeSphere(a.n)
    grid, lat, lon = read_asc(a.asc)
    w = np.cos(np.radians(lat))[:, None]*np.ones((1, lon.size))
    ok = np.isfinite(grid)
    print(f'GlobSed grid {grid.shape}: valid area {w[ok].sum()/w.sum():.3f}, mean {(np.where(ok, grid, 0)*w).sum()/w[ok].sum():.1f} m')
    sed, share = node_means(g, grid, lat, lon)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/f'globsed-n{a.n}.npz', sediment_m=sed.astype(np.float32), valid_share=share.astype(np.float32),
                        n=a.n, source=str(a.asc))
    v = share > .5
    print(f'n = {a.n}: nodes with > half their pixels valid {g.area[v].sum()/g.area.sum():.3f} of the surface, mean there '
          f'{np.average(sed[v], weights=g.area[v]):.1f} m')


if __name__ == '__main__':
    main()
