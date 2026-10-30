"""Earth's ocean-floor age (Seton et al. 2020, EarthByte age grid 2020.1) on the planet layer's grid: seafloor
production, the age distribution, and GlobSed's sediment thickness by floor age (plan doc, thirty-seventh pass).

  python -m scripts.earth_seafloor_age          # writes outputs/planet-reference/earth-seafloor-age-n217.npz / .json

Source: data/EarthByte/Grids/age.2020.1.GTS2012.2m.grd (classic netCDF, CDF1: read here with numpy, so the project's
.venv needs neither h5py nor netCDF4; the .nc versions are netCDF-4 / HDF5). Ages in Myr (GTS2012 time scale), 2 arc
minute gridline-registered grid, NaN off the dated ocean floor. Read every third row and column (6 arc minutes).

Printed: dated ocean-floor area and its mean / median age; production as area younger than A / A for A = 2 ... 60 Myr
(steady production would give the same value for every A; subduction of young floor lowers the long windows); area by
20 Myr bins; on CubeSphere(217) nodes (more than half their pixels dated), GlobSed bulk thickness (scripts.earth_sediment)
by 10 Myr age bins, mean and median.

References: Seton, M. et al. (2020), G3 21, e2020GC009214; Straume, E. O. et al. (2019), G3 20, 1756-1772 (GlobSed).
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

from planet_layer.sphere_graph import CubeSphere
from scripts import earth_sediment as es

ROOT = Path(__file__).resolve().parents[1]
AGE_GRID = ROOT/'data'/'EarthByte'/'Grids'/'age.2020.1.GTS2012.2m.grd'
OUT = ROOT/'outputs'/'planet-reference'
_TYPES = {1: ('b', 1), 2: ('c', 1), 3: ('>i2', 2), 4: ('>i4', 4), 5: ('>f4', 4), 6: ('>f8', 8)}


def read_cdf(path):
    """Variables of a classic netCDF file (CDF1 or CDF2), as {name: (dims, dtype, offset)} with dims {name: length}."""
    f = open(path, 'rb')
    head = f.read(1 << 20)
    f.close()
    assert head[:3] == b'CDF' and head[3] in (1, 2), 'not a classic netCDF file'
    big = head[3] == 2
    pos = 8

    def i4():
        nonlocal pos
        v = struct.unpack('>i', head[pos:pos+4])[0]; pos += 4
        return v

    def name():
        nonlocal pos
        n = i4(); s = head[pos:pos+n].decode(); pos += (n+3)//4*4
        return s

    def atts():
        nonlocal pos
        i4(); n = i4()
        for _ in range(n):
            name(); t = i4(); m = i4()
            pos += (m*_TYPES[t][1]+3)//4*4
    i4(); nd = i4()
    dims = [(name(), i4()) for _ in range(nd)]
    atts()
    i4(); nv = i4()
    out = {}
    for _ in range(nv):
        vn = name()
        ids = [i4() for _ in range(i4())]
        atts()
        t = i4(); i4()
        if big:
            off = struct.unpack('>q', head[pos:pos+8])[0]; pos += 8
        else:
            off = i4()
        out[vn] = ([dims[k] for k in ids], _TYPES[t][0], off)
    return out


def read_grid(path=AGE_GRID, step=3):
    """(age grid rows by latitude, lat, lon), every step-th row and column, the duplicate 180-degree column dropped."""
    v = read_cdf(path)

    def arr(nm):
        dims, dt, off = v[nm]
        return np.memmap(path, dtype=dt, mode='r', offset=off, shape=tuple(n for _, n in dims))
    lat, lon, z = np.array(arr('lat')), np.array(arr('lon')), arr('z')
    z = np.array(z[::step, ::step], np.float32)
    lat, lon = lat[::step], lon[::step]
    if np.isclose(lon[-1]-lon[0], 360.):
        z, lon = z[:, :-1], lon[:-1]
    return z, lat, lon


def summary(z, lat, lon):
    R = 6371.0
    w = (R**2*np.radians(abs(lat[1]-lat[0]))*np.radians(abs(lon[1]-lon[0]))*np.cos(np.radians(lat)))[:, None]*np.ones((1, lon.size))
    w[[0, -1], :] *= .5                                  # gridline-registered: half cells at the poles
    ok = np.isfinite(z)
    A = float(w[ok].sum())
    r = {'dated_area_mkm2': A/1e6, 'dated_share': A/float(w.sum()), 'mean_age_myr': float(np.average(z[ok], weights=w[ok])),
         'median_age_myr': float(np.median(z[ok]))}
    r['production_km2_yr'] = {str(a): float(w[ok & (z < a)].sum()/a/1e6) for a in (2, 5, 10, 20, 40, 60)}
    edges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 400]
    r['area_share_by_age'] = {f'{lo}-{hi}': float(w[ok & (z >= lo) & (z < hi)].sum()/A) for lo, hi in zip(edges[:-1], edges[1:])}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=217)
    ap.add_argument('--grid', default=str(AGE_GRID))
    a = ap.parse_args(argv)
    z, lat, lon = read_grid(a.grid)
    r = summary(z, lat, lon)
    print(f"Seton et al. 2020 age grid: dated ocean floor {r['dated_area_mkm2']:.1f} Mkm^2 ({r['dated_share']:.3f} of the "
          f"surface), mean age {r['mean_age_myr']:.1f} Myr, median {r['median_age_myr']:.1f}")
    print('  production (area younger than A / A, km^2/yr): '+', '.join(f'A {k}: {v:.2f}' for k, v in r['production_km2_yr'].items()))
    print('  area share by age (Myr): '+', '.join(f'{k} {v:.3f}' for k, v in r['area_share_by_age'].items()))
    g = CubeSphere(a.n)
    age, share = es.node_means(g, z, lat, lon)
    sed, sshare = es.load(a.n)
    ok = (share > .5) & (sshare > .5)
    rows = {}
    for lo in range(0, 180, 10):
        m = ok & (age >= lo) & (age < lo+10)
        if m.sum() > 20:
            rows[f'{lo}-{lo+10}'] = {'area_mkm2': float(g.area[m].sum()/1e12), 'mean_m': float(np.average(sed[m], weights=g.area[m])),
                                     'median_m': float(np.median(sed[m]))}
    r['globsed_by_age'] = rows
    print(f'  GlobSed (bulk) by floor age on n = {a.n} (nodes > half dated and > half GlobSed-valid):')
    for k, v in rows.items():
        print(f"    {k:>8} Myr: area {v['area_mkm2']:5.1f} Mkm^2, mean {v['mean_m']:6.0f} m, median {v['median_m']:6.0f} m")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/f'earth-seafloor-age-n{a.n}.npz', age_myr=age.astype(np.float32), dated_share=share.astype(np.float32),
                        n=a.n, source=str(a.grid))
    r['source'] = 'Seton et al. 2020 age grid 2020.1 (GTS2012, 2 arc min, read every 3rd sample); GlobSed v3 (scripts.earth_sediment)'
    (OUT/f'earth-seafloor-age-n{a.n}.json').write_text(json.dumps(r, indent=1))
    print(f'written {OUT}/earth-seafloor-age-n{a.n}.npz / .json')


if __name__ == '__main__':
    main()
