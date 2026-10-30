"""The erosion laws of the planet chain against Earth's measured erosion rates (plan doc, fourteenth pass).

  python -m scripts.erosion_law_check --grid 217
  python -m scripts.erosion_law_check --grid 868

Each law is applied once, at its own grid and time step, to Earth's present surface on that grid (ETOPO,
outputs/planet-reference/earth-n<n>.npz):
  217: the plate layer's coarse erosion (planet_layer.plates.PARAMS: K factor, Sc, Kd factor, theta; 250 kyr);
  868: the landscape stage (planet_training.landscape_evolution.DEFAULTS, theta = run_planet_landscape.THETA; 50 kyr).
Erodibility as in the planet: 1 in young orogens, `stable` elsewhere; Earth's young orogens are Bird 2003's orogen
zones (data/tectonicplates-master/GeoJSON/PB2002_orogens.json).

The law's rate is its gross erosion (rock removed, m/Myr; deposition is not subtracted, as a cosmogenic basin average
measures the lowering of the eroding surface). It is averaged over every OCTOPUS v2 basin (Codilean et al. 2022,
CRN Global collection, data/octopus/crn_int_basins.json, EPSG:4326) by sampling the basin polygon on a ~2 km
lattice and reading the nearest node. Only basins at least `min_cells` grid cells in area are scored (the law
predicts a mean over its own cells; small steep catchments are not comparable).

Rule, fixed before the data is read (fourteenth pass):
  ratio = law / measured per basin; classes by measured rate: < 20, 20-100, 100-500, >= 500 m/Myr (= mm/kyr);
  a class with at least 20 basins passes if its median log10 ratio (the law's bias) lies within +-0.2 (a factor
  1.6); every scored class must pass. Spearman correlation is reported.
Writes outputs/erosion-law-check/<grid>-<timestamp>/results.json and scatter.png.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ((0., 20.), (20., 100.), (100., 500.), (500., 1e9))
BIAS_BAND = .2          # |median log10(law / measured)| per class
MIN_BASINS = 20
MIN_CELLS = 1.          # basin area >= this many grid cells
RATE_FIELDS = ('EBE_MMKYR', 'ebe_mmkyr', 'EBE_MM_KYR', 'ebe_mm_kyr')
AREA_FIELDS = ('AREA', 'area', 'AREA_KM2', 'area_km2')


def lonlat(xyz):
    x, y, z = np.asarray(xyz).T
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1, 1)))


def to_xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)), -1)


def orogen_mask(g, path):
    """Nodes inside Bird 2003's orogen polygons (longitudes unwrapped around each ring's first vertex)."""
    from matplotlib.path import Path as MPath
    lon, lat = lonlat(g.xyz)
    inside = np.zeros(g.size, bool)
    for f in json.loads(Path(path).read_text(encoding='utf-8'))['features']:
        geom = f['geometry']
        for poly in (geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]):
            ring = np.asarray(poly[0], float)
            l0 = ring[0, 0]
            rl = (ring[:, 0]-l0+180) % 360-180+l0
            pl = (lon-l0+180) % 360-180+l0
            inside |= MPath(np.c_[rl, ring[:, 1]]).contains_points(np.c_[pl, lat])
    return inside


def law_rate(n, z, orogen):
    """Gross erosion (m/Myr) of the chain's law at grid n on surface z, one model step."""
    from planet_layer import graph_landscape as gl
    from planet_layer.sphere_graph import CubeSphere
    g = CubeSphere(n)
    if n == 217:
        from planet_layer.plates import PARAMS as p
        lp = {**gl.DEFAULTS, 'K': gl.DEFAULTS['K']*p['erosion_K_factor'], 'Sc': p['erosion_Sc'],
              'Kd': gl.DEFAULTS['Kd']*p['erosion_Kd_factor'], 'open_edges': False, 'theta': p['erosion_theta']}
        stable = p['stable_erodibility']
        dt = p['dt_myr']*1e6/p['erosion_substeps']
    else:
        from scripts.run_planet_landscape import THETA
        lp = {**gl.DEFAULTS, 'open_edges': False, 'theta': THETA}
        stable = .15                                  # planet_layer.coupling.erodibility
        dt = gl.DEFAULTS['dt']
    K = lp['K']*np.where(orogen, 1., stable)
    r = gl.surface_step(g, np.asarray(z, np.float64), z < -1500., None, np.ones(g.size, bool), K, lp, dt,
                        g.spacing_min(), g.diffusion_bound(), keep_deep=True)
    return g, r['eroded']/dt*1e6, lp, dt


def basin_samples(geom, spacing_km=2.):
    """Points (lon, lat) on a lattice inside a basin polygon; at least its vertex mean for tiny basins."""
    from matplotlib.path import Path as MPath
    pts = []
    for poly in (geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]):
        ring = np.asarray(poly[0], float)
        lo0, lo1 = ring[:, 0].min(), ring[:, 0].max(); la0, la1 = ring[:, 1].min(), ring[:, 1].max()
        dla = spacing_km/111.2; dlo = dla/max(np.cos(np.radians((la0+la1)/2)), .05)
        LO, LA = np.meshgrid(np.arange(lo0, lo1+dlo, dlo), np.arange(la0, la1+dla, dla))
        q = np.c_[LO.ravel(), LA.ravel()]
        m = MPath(ring).contains_points(q)
        pts.append(q[m] if m.any() else ring[:-1].mean(0, keepdims=True))
    return np.concatenate(pts)


def field(props, names):
    for k in names:
        if k in props and props[k] is not None:
            return float(props[k])
    return np.nan


def compare(g, rate, basins_json, min_cells=MIN_CELLS):
    """basins_json: a GeoJSON path or the parsed FeatureCollection."""
    from planet_layer import fields
    cell_km2 = g.area.mean()/1e6
    rows = []
    data = basins_json if isinstance(basins_json, dict) else json.loads(Path(basins_json).read_text(encoding='utf-8'))
    for f in data['features']:
        pr = f['properties']
        meas, area = field(pr, RATE_FIELDS), field(pr, AREA_FIELDS)
        if not np.isfinite(meas) or meas <= 0 or f.get('geometry') is None:
            continue
        pts = basin_samples(f['geometry'])
        if not np.isfinite(area):                    # area from the lattice when the attribute is missing
            area = len(pts)*4.
        if area < min_cells*cell_km2:
            continue
        nodes = fields.nearest_nodes(g, to_xyz(pts[:, 0], pts[:, 1]))
        rows.append({'measured_m_myr': meas, 'law_m_myr': float(rate[nodes].mean()), 'area_km2': area,
                     'lon': float(pts[:, 0].mean()), 'lat': float(pts[:, 1].mean())})
    return rows


def score(rows):
    meas = np.array([r['measured_m_myr'] for r in rows]); law = np.array([r['law_m_myr'] for r in rows])
    lr = np.log10(np.maximum(law, 1e-3)/meas)
    out = {'basins': len(rows), 'classes': {}, 'passes': True}
    for lo, hi in CLASSES:
        m = (meas >= lo) & (meas < hi)
        c = {'n': int(m.sum())}
        if m.sum():
            c.update({'median_log10_ratio': float(np.median(lr[m])), 'law_median_m_myr': float(np.median(law[m])),
                      'measured_median_m_myr': float(np.median(meas[m]))})
        c['scored'] = bool(m.sum() >= MIN_BASINS)
        c['passes'] = bool(not c['scored'] or abs(c['median_log10_ratio']) <= BIAS_BAND)
        out['passes'] &= c['passes']
        out['classes'][f'{lo:g}-{hi:g}'] = c
    out['passes'] &= any(c['scored'] for c in out['classes'].values())   # nothing scored is not a pass
    if len(rows) > 2:
        rk = lambda v: np.argsort(np.argsort(v))
        out['spearman'] = float(np.corrcoef(rk(meas), rk(law))[0, 1])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', type=int, choices=(217, 868), default=217)
    p.add_argument('--basins', default=str(ROOT/'data'/'octopus'/'crn_int_basins.json'))
    p.add_argument('--orogens', default=str(ROOT/'data'/'tectonicplates-master'/'GeoJSON'/'PB2002_orogens.json'))
    p.add_argument('--min-cells', type=float, default=MIN_CELLS)
    p.add_argument('--out', default=None)
    a = p.parse_args(argv)
    from planet_layer.sphere_graph import CubeSphere
    z = np.load(ROOT/'outputs'/'planet-reference'/f'earth-n{a.grid}.npz')['elevation_m'].astype(np.float64)
    g0 = CubeSphere(a.grid)
    oro = orogen_mask(g0, a.orogens)
    g, rate, lp, dt = law_rate(a.grid, z, oro)
    rows = compare(g, rate, a.basins, a.min_cells)
    if not rows:
        f0 = json.loads(Path(a.basins).read_text(encoding='utf-8'))['features'][0]
        raise SystemExit(f'no basin scored: rate field {RATE_FIELDS} or area field {AREA_FIELDS} not found, or all basins '
                         f'smaller than {a.min_cells:g} cells. First feature properties: {sorted(f0["properties"])}')
    res = score(rows)
    res.update({'grid': a.grid, 'law': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in lp.items()},
                'dt_yr': dt, 'min_cells': a.min_cells, 'rule': {'bias_band_log10': BIAS_BAND, 'min_basins': MIN_BASINS}})
    out = Path(a.out) if a.out else ROOT/'outputs'/'erosion-law-check'/f"{a.grid}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps({**res, 'rows': rows}, indent=1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meas = np.array([r['measured_m_myr'] for r in rows]); law = np.array([r['law_m_myr'] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.loglog(meas, np.maximum(law, 1e-2), '.', ms=3, alpha=.5)
    lim = [1e-1, 1e4]
    ax.plot(lim, lim, 'k-', lw=.8)
    ax.plot(lim, [l*10**BIAS_BAND for l in lim], 'k:', lw=.6); ax.plot(lim, [l/10**BIAS_BAND for l in lim], 'k:', lw=.6)
    ax.set_xlabel('measured (10Be, OCTOPUS), m/Myr'); ax.set_ylabel(f'law at n={a.grid}, m/Myr')
    ax.set_title(f'n={a.grid}: {len(rows)} basins; passes {res["passes"]}', fontsize=9)
    fig.tight_layout(); fig.savefig(out/'scatter.png', dpi=110); plt.close(fig)
    print(f"n={a.grid}: {len(rows)} basins >= {a.min_cells:g} cells; spearman {res.get('spearman', float('nan')):.2f}")
    for k, c in res['classes'].items():
        if c['n']:
            print(f"  measured {k:>9} m/Myr: n {c['n']:4d}, law / measured median x{10**c['median_log10_ratio']:.2f} "
                  f"(law {c['law_median_m_myr']:.0f}, measured {c['measured_median_m_myr']:.0f}){'' if c['scored'] else ' (too few to score)'}"
                  f"{'' if c['passes'] else '  FAIL'}")
    print(f"passes: {res['passes']}; written {out}")
    return res


if __name__ == '__main__':
    main()
