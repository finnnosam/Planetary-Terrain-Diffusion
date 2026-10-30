"""Where the planet's thick continental interior comes from (twenty-third pass, item 2 of the 25 Oct handoff).

  python -m scripts.interior_probe outputs\\planet-calibration\\sweep-20261024-173200\\rift_rule-pure_shear
  python -m scripts.interior_probe --latest            the newest sweep, with 3-seed means and the item-2 tests

Read-only. Continental crust read label-free (crust + sediment >= 20 km, crust_review). For the present day and for
the start (t = -250 Myr) of every seed: crystalline crust thickness inland from passive edges (the margin_width
bins), split by provenance:
  inherited   crust older than the run that never took part in a collision during the run (the supercontinent's
              cratons and old belts as initialised)
  collided    crust whose orogeny clock was reset during the run (orogeny age < run length): collision contacts,
              and with orogen_rule 'belt' / 'strain' also belts, collapsing or deforming crust
  new         crust born during the run (arcs that became continental, rift gaps)
and the share of the continent thicker than 45 / 55 km by provenance and by distance to the nearest plate boundary.
Also, present day: the land carrying young-orogen erodibility (>= 0.5, as the erosion law's calibration counts Bird's
orogens at 1) and the land deforming at |tectonic thickening| / thickness above 1, 3, 10 nanostrain/yr, against Earth
(Bird 2003 orogens: 16% of land, 76% of land above 2 km). Earth side (--earth): CRUST1.0 crystalline thickness by crust
type group in the same inland bins, and Earth's thick crust and high land inside Bird's orogens.
E references (twenty-sixth pass), both sides: steepest-descent slope of land above 2 km (`slope_m_km`), and the depth
of the > 45 km crust from its edge (`edge_depth_km`, graph distance to the nearest node outside, area-weighted), with
its connected pieces. Reported beside the item-2 tests in the 3-seed summary (E tests, not counted in the 5).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import graph_landscape as gl
from planet_layer.sphere_graph import CubeSphere
from scripts.crust_review import column_thickness_m, label_free_continent, run_params, wpct
from scripts.margin_width import BOUNDARY_KM, C1, ROOT, crust1_cell, earth_boundaries

# E references measured 25 Oct (late) (ETOPO / CRUST1.0 on CubeSphere(217), no ice sheets): slope of land above 2 km p50
# 11.1 m/km, share under 5 m/km 0.23; > 45 km crust depth from its edge p50 / p90 92 / 322 km
EARTH_E = {'slope_p50': 11.1, 'slope_lt5': .23, 'depth_p50': 92., 'depth_p90': 322.}
BINS = ((0, 150), (150, 200), (200, 300), (300, 500), (500, 1000), (1000, 5000))
EARTH_CRYST = {'150-200': 31.6, '200-300': 33.8, '300-500': 35.4, '500-1000': 36.6}     # twenty-second pass


def slope_m_km(g, e):
    """Steepest-descent slope (m/km) on every node: the largest drop to a neighbour over the true distance, >= 0."""
    nb = g.nbr; ok = nb >= 0
    s = np.where(ok, (e[:, None]-e[np.where(ok, nb, 0)])/np.where(ok, g.dist, 1.), -np.inf).max(1)
    return np.clip(s, 0, None)*1e3


def edge_depth_km(g, mask):
    """Graph distance (km) from every node of mask to the nearest node outside it (a one-node belt reads one spacing),
    and the areas (m^2) of mask's connected pieces, largest first."""
    nb = g.nbr
    d = gl.distance_to(~mask, nb, g.dist, 5000e3)/1e3
    lab = np.full(g.size, -1); pieces = []
    for s0 in np.flatnonzero(mask):
        if lab[s0] >= 0:
            continue
        st = [s0]; lab[s0] = len(pieces); A = 0.
        while st:
            i = st.pop(); A += g.area[i]
            for k in nb[i]:
                if k >= 0 and mask[k] and lab[k] < 0:
                    lab[k] = len(pieces); st.append(k)
        pieces.append(A)
    return d, np.sort(np.array(pieces))[::-1]


def e_refs(g, e, land, thick):
    """The E references on one surface: slope of land above 2 km, depth of the thick crust from its edge, its pieces."""
    a = g.area; s = slope_m_km(g, e); hi = land & (e > 2000)
    d, pieces = edge_depth_km(g, thick)
    return {'slope_p25': wpct(s[hi], a[hi], 25), 'slope_p50': wpct(s[hi], a[hi], 50), 'slope_p75': wpct(s[hi], a[hi], 75),
            'slope_lt5': float(a[hi & (s < 5)].sum()/max(a[hi].sum(), 1.)), 'land_slope_p50': wpct(s[land], a[land], 50),
            'depth_p50': wpct(d[thick], a[thick], 50) if thick.any() else np.nan,
            'depth_p90': wpct(d[thick], a[thick], 90) if thick.any() else np.nan,
            'pieces': len(pieces), 'largest_mkm2': [float(x/1e12) for x in pieces[:4]]}


def show_e(r):
    print(f"    E: slope of land > 2 km p25 / 50 / 75 {r['slope_p25']:.1f} / {r['slope_p50']:.1f} / {r['slope_p75']:.1f} m/km, "
          f"share under 5 m/km {r['slope_lt5']:.2f} (Earth p50 {EARTH_E['slope_p50']}, {EARTH_E['slope_lt5']}); all land p50 "
          f"{r['land_slope_p50']:.1f}; > 45 km depth from its edge p50 / p90 {r['depth_p50']:.0f} / {r['depth_p90']:.0f} km "
          f"(Earth {EARTH_E['depth_p50']:.0f} / {EARTH_E['depth_p90']:.0f}), {r['pieces']} pieces, largest "
          + ' / '.join(f'{x:.1f}' for x in r['largest_mkm2']) + ' Mkm^2 (Earth 7.9 / 1.4 / 1.1 / 0.5)')
    if 'ext_inland_share' in r:
        print(f"    extended by E: {r['ext_share_of_continent']:.3f} of the continent; {r['ext_inland_share']:.3f} of the 150-1000 km "
              f"inland bins at {r['ext_inland_med']:.1f} km (Earth's extended crust 0.09-0.11 at 30.7-31.7)")


def read(g, snap, p, run_myr):
    z = np.load(snap)
    Tcol = column_thickness_m(z, p); Tc = z['crust_thickness_m'].astype(float)
    cont = label_free_continent(Tcol)
    nb = g.nbr; ok = nb >= 0; a = g.area
    ocean_nb = np.any(np.where(ok, ~cont[np.where(ok, nb, 0)], False), axis=1)
    bnd = z['boundary_type'] != 0
    dbnd = gl.distance_to(bnd, nb, g.dist, 5000e3)/1e3
    passive = cont & ocean_nb & (dbnd > BOUNDARY_KM)
    dp = gl.distance_to(passive, nb, g.dist, 5000e3)/1e3
    dcoast = gl.distance_to(cont & ocean_nb, nb, g.dist, 5000e3)/1e3
    oa = z['orogeny_age_myr'].astype(float); ca = z['crust_age_myr'].astype(float)
    collided = np.isfinite(oa) & (oa < run_myr)
    new = ca < run_myr
    prov = {'inherited': cont & ~collided & ~new, 'collided': cont & collided & ~new, 'new': cont & new}
    out = {'area': float(a[cont].sum()/a.sum()), 'profile': {}, 'coast_profile': {}}
    for name, d, extra in (('profile', dp, dbnd > BOUNDARY_KM), ('coast_profile', dcoast, np.ones(g.size, bool))):
        for lo, hi in BINS:
            m = cont & extra & (d >= lo) & (d < hi)
            if m.sum() < 10:
                continue
            row = {'med': wpct(Tc[m], a[m], 50)/1e3, 'share_of_continent': float(a[m].sum()/a[cont].sum())}
            for k, pm in prov.items():
                mm = m & pm
                row[k] = (float(a[mm].sum()/a[m].sum()), wpct(Tc[mm], a[mm], 50)/1e3 if mm.sum() > 5 else np.nan)
            out[name][f'{lo}-{hi}'] = row
    for lim in (45e3, 55e3):
        th = cont & (Tcol > lim)
        out[f'gt{lim/1e3:.0f}'] = float(a[th].sum()/a[cont].sum())
        out[f'gt{lim/1e3:.0f}_by'] = {k: float(a[th & pm].sum()/a[cont].sum()) for k, pm in prov.items()}
        out[f'gt{lim/1e3:.0f}_near_boundary'] = float(a[th & (dbnd < 500)].sum()/max(a[th].sum(), 1))
    out['median_cryst_by'] = {k: (float(a[pm].sum()/a[cont].sum()), wpct(Tc[pm], a[pm], 50)/1e3 if pm.sum() > 5 else np.nan)
                              for k, pm in prov.items()}
    out['median_cryst'] = wpct(Tc[cont], a[cont], 50)/1e3
    e = z['elevation_m'].astype(float); land = e > 0; hi = e > 2000
    flag = z['continental'].astype(bool)
    since = np.where(np.isfinite(oa), np.clip(oa, 0, None), np.inf)
    erod = np.where(flag, p['stable_erodibility']+(1-p['stable_erodibility'])*np.exp(-since/p['orogen_erodibility_tau_myr']), 1.)
    young = erod >= .5
    th = cont & (Tcol > 45e3)
    out['land_gt2km'] = float(a[hi].sum()/a[land].sum())
    out['young_orogen'] = {'land': float(a[land & young].sum()/a[land].sum()),
                           'land_gt2km': float(a[hi & young].sum()/max(a[hi].sum(), 1.)),
                           'gt45': float(a[th & young].sum()/max(a[th].sum(), 1.)),
                           'gt45_mean_erodibility': float(np.average(erod[th], weights=a[th])) if th.any() else np.nan}
    rate = np.abs(z['tectonic_thickening_m'].astype(float))/np.maximum(Tc, 1e3)/(p['dt_myr']*1e6)
    out['deforming'] = {f'{c*1e9:.0f}': {'land': float(a[land & (rate > c)].sum()/a[land].sum()),
                                         'land_gt2km': float(a[hi & (rate > c)].sum()/max(a[hi].sum(), 1.)),
                                         'gt45': float(a[th & (rate > c)].sum()/max(a[th].sum(), 1.))} for c in (1e-9, 3e-9, 1e-8)}
    out['gt45_land_elev_p50_m'] = float(np.median(e[th & land])) if (th & land).any() else np.nan
    out['E'] = e_refs(g, e, land, th)
    if 'extension_age_myr' in z.files:               # crust thinned or made by extension E (twenty-seventh pass)
        ext = cont & np.isfinite(z['extension_age_myr'])
        m = cont & (dbnd > BOUNDARY_KM) & (dp >= 150) & (dp < 1000)
        out['E']['ext_inland_share'] = float(a[m & ext].sum()/max(a[m].sum(), 1.))
        out['E']['ext_inland_med'] = wpct(Tc[m & ext], a[m & ext], 50)/1e3 if (m & ext).sum() > 5 else np.nan
        out['E']['ext_share_of_continent'] = float(a[ext].sum()/a[cont].sum())
    return out


def show(tag, r):
    print(f"  {tag}: continent {r['area']:.3f}, crystalline median {r['median_cryst']:.1f} km; by provenance (share, median): "
          + ', '.join(f"{k} {v[0]:.2f} / {v[1]:.1f}" for k, v in r['median_cryst_by'].items()))
    print(f"    > 45 km {r['gt45']:.3f} (inherited / collided / new {r['gt45_by']['inherited']:.3f} / {r['gt45_by']['collided']:.3f} / "
          f"{r['gt45_by']['new']:.3f}; {r['gt45_near_boundary']:.0%} within 500 km of a boundary), > 55 km {r['gt55']:.3f} "
          f"({r['gt55_near_boundary']:.0%} near a boundary)")
    if 'young_orogen' in r:
        y = r['young_orogen']
        print(f"    land > 2 km {r['land_gt2km']:.2f} of land (Earth 0.05); young-orogen erodibility on land {y['land']:.2f} (Earth, Bird "
              f"orogens 0.16), on land > 2 km {y['land_gt2km']:.2f} (0.76), on > 45 km crust {y['gt45']:.2f} (mean erodibility "
              f"{y['gt45_mean_erodibility']:.2f}); > 45 km land elevation p50 {r['gt45_land_elev_p50_m']:.0f} m (Earth 1,234)")
        print('    deforming (nanostrain/yr: land / land > 2 km / > 45 km crust): ' + ', '.join(
            f"{k}: {v['land']:.2f} / {v['land_gt2km']:.2f} / {v['gt45']:.2f}" for k, v in r['deforming'].items()))
    if 'E' in r:
        show_e(r['E'])
    for name in ('profile', 'coast_profile'):
        print(f"    {'from passive edges' if name == 'profile' else 'from any coast'} (km: crystalline median [Earth]; inherited / collided / new share, median):")
        for k, v in r[name].items():
            e = f" [{EARTH_CRYST[k]:.1f}]" if name == 'profile' and k in EARTH_CRYST else ''
            print(f"      {k:>9}: {v['med']:.1f}{e} ({v['share_of_continent']:.0%} of continent); "
                  + ', '.join(f"{p} {v[p][0]:.2f} / {v[p][1]:.1f}" for p in ('inherited', 'collided', 'new')))


GROUPS = {'Precambrian/platform': ['D-', 'E-', 'F-', 'G1', 'G2', 'H1', 'H2', 'I1', 'I2'],
          'shelf/slope': ['C-', 'S-', 'W-', 'V2', 'Y1', 'Y2', 'U-'], 'extended': ['M-', 'N-', 'T-'],
          'orogen/arc': ['J-', 'K-', 'L1', 'L2', 'O-', 'P-', 'Q-', 'R1', 'R2', 'X-'], 'other': ['Z1', 'Z2']}


def earth(g):
    """Earth (CRUST1.0 x ETOPO on the same graph): crystalline thickness by crust type group inland from passive
    edges; thick crust and high land inside Bird's orogens."""
    from scripts.erosion_law_check import orogen_mask
    a = g.area; nb = g.nbr; ok = nb >= 0
    cell = crust1_cell(g)
    T = np.loadtxt(C1/'crsthk.xyz')[:, 2][cell]*1e3; S = np.loadtxt(C1/'sedthk.xyz')[:, 2][cell]*1e3; Tc = T-S
    ty = np.array([l.split() for l in (C1/'CNtype1-1.txt').read_text().splitlines() if l.strip()]).ravel()[cell]
    grp = np.full(g.size, 'other', object)
    for k, v in GROUPS.items():
        grp[np.isin(ty, v)] = k
    cont = label_free_continent(T)
    ocean_nb = np.any(np.where(ok, ~cont[np.where(ok, nb, 0)], False), axis=1)
    dbnd = gl.distance_to(earth_boundaries(g), nb, g.dist, 5000e3)/1e3
    dp = gl.distance_to(cont & ocean_nb & (dbnd > BOUNDARY_KM), nb, g.dist, 5000e3)/1e3
    print('Earth (CRUST1.0, total >= 20 km): crystalline p10 / 25 / 50 / 75 / 90 by crust type group (share of continent)')
    for k in GROUPS:
        m = cont & (grp == k)
        print(f"  {k:>20}: {' / '.join(f'{wpct(Tc[m], a[m], q)/1e3:.1f}' for q in (10, 25, 50, 75, 90))} km ({a[m].sum()/a[cont].sum():.2f})")
    m = cont & (grp == 'Precambrian/platform') & (dbnd > 500)
    print(f"  Precambrian/platform > 500 km from plate boundaries: {' / '.join(f'{wpct(Tc[m], a[m], q)/1e3:.1f}' for q in (10, 25, 50, 75, 90))} km")
    print('  inland from passive edges (km: crystalline p50; group share / p50):')
    for lo, hi in BINS[1:]:
        m = cont & (dbnd > BOUNDARY_KM) & (dp >= lo) & (dp < hi)
        print(f"    {lo}-{hi}: {wpct(Tc[m], a[m], 50)/1e3:.1f}; " + ', '.join(
            f"{k} {a[m & (grp == k)].sum()/a[m].sum():.2f} / {wpct(Tc[m & (grp == k)], a[m & (grp == k)], 50)/1e3:.1f}"
            for k in GROUPS if (m & (grp == k)).sum() > 5))
    oro = orogen_mask(g, ROOT/'data'/'tectonicplates-master'/'GeoJSON'/'PB2002_orogens.json')
    e = np.load(ROOT/'outputs'/'planet-reference'/'earth-n217.npz')['elevation_m'].astype(float)
    land = e > 0; hi = e > 2000
    print(f"  Bird orogens: {a[land & oro].sum()/a[land].sum():.2f} of land, {a[hi & oro].sum()/a[hi].sum():.2f} of land > 2 km "
          f"(land > 2 km: {a[hi].sum()/a[land].sum():.3f} of land)")
    for lim in (45e3, 50e3, 55e3):
        th = cont & (T > lim)
        print(f"  > {lim/1e3:.0f} km crust: {a[th].sum()/a[cont].sum():.3f} of continent, {a[th & oro].sum()/a[th].sum():.2f} inside "
              f"Bird orogens; elevation p50 {np.median(e[th]):.0f} m (outside orogens {np.median(e[th & ~oro]):.0f} m)")
    from scripts.earth_reference import ice_mask
    ice = ice_mask(g)
    print('  Earth, no ice sheets:'); show_e(e_refs(g, e, land & ~ice, cont & (T > 45e3) & ~ice))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--earth', action='store_true', help='also the Earth side (CRUST1.0, ETOPO, Bird orogens)')
    ap.add_argument('--latest', action='store_true', help='the newest outputs/planet-calibration/sweep-* folder')
    a = ap.parse_args(argv)
    if a.latest:
        a.dirs = [str(sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1])]
        print('sweep', a.dirs[0])
    g = CubeSphere(217)
    if a.earth:
        earth(g)
    seeds = []
    for d in map(Path, a.dirs):
        seeds += [d] if (d/'snapshots').is_dir() else sorted(x for x in d.rglob('seed*') if (x/'snapshots').is_dir())
    res = {}
    for s in seeds:
        p = run_params(s)
        snaps = sorted((s/'snapshots').glob('t*.npz'), key=lambda f: float(f.stem[1:]))
        t0 = float(snaps[0].stem[1:]); run = -t0
        print(f"{s.parent.name}/{s.name}")
        res[str(s)] = {}
        for snap in (snaps[0], s/'snapshots'/'t+0000.0.npz'):
            r = read(g, snap, p, run+1e-6 if snap != snaps[0] else 0.)
            res[str(s)][snap.stem] = r
            show(snap.stem, r)
    summary(res)
    return res


# the item-2 tests (twenty-third pass, fixed 25 Oct), read on 3-seed means; the elevation tests and the export band
# are read from scripts.elevation_check
TESTS = {'profile within 1.5 km of Earth': None, 'gt45 <= 0.13': (0., .13), 'young orogen on land 0.10-0.25': (.10, .25),
         'young orogen on land > 2 km >= 0.5': (.5, 1.), 'land > 2 km <= 0.10': (0., .10)}

SHORT = {'gt45 <= 0.13': '> 45 km', 'young orogen on land 0.10-0.25': 'young orogen / land',
         'young orogen on land > 2 km >= 0.5': 'young orogen / land > 2 km', 'land > 2 km <= 0.10': 'land > 2 km'}


def summary(res):
    combos = {}; seeds_of = {}
    for s, r in res.items():
        if 't+0000.0' in r:
            combos.setdefault(Path(s).parent.name, []).append(r['t+0000.0'])
            seeds_of.setdefault(Path(s).parent.name, []).append(s)
    if not combos:
        return
    print('3-seed means (item-2 tests; the crust median is read from crust_review):')
    for name, rs in combos.items():
        prof = {k: float(np.nanmean([r['profile'][k]['med'] if k in r['profile'] else np.nan for r in rs])) for k in EARTH_CRYST}
        dev = max(abs(prof[k]-EARTH_CRYST[k]) for k in EARTH_CRYST)
        v = {'gt45 <= 0.13': np.mean([r['gt45'] for r in rs]),
             'young orogen on land 0.10-0.25': np.mean([r['young_orogen']['land'] for r in rs]),
             'young orogen on land > 2 km >= 0.5': np.mean([r['young_orogen']['land_gt2km'] for r in rs]),
             'land > 2 km <= 0.10': np.mean([r['land_gt2km'] for r in rs])}
        ok = [dev <= 1.5]+[TESTS[k][0] <= x <= TESTS[k][1] for k, x in v.items()]
        print(f"  {name} ({len(rs)} seeds): passes {sum(ok)} of 5 | profile " + ' / '.join(f"{prof[k]:.1f}" for k in EARTH_CRYST)
              + f" (worst {dev:+.1f} km) {'ok' if ok[0] else 'x'}; " + '; '.join(f"{SHORT[k]} {x:.3f} {'ok' if o else 'x'}" for (k, x), o in zip(v.items(), ok[1:])))
        E = {k: float(np.mean([r['E'][k] for r in rs])) for k in ('slope_p50', 'slope_lt5', 'depth_p50', 'depth_p90')}
        if all('ext_inland_share' in r['E'] for r in rs):
            sh = np.mean([r['E']['ext_inland_share'] for r in rs]); md = np.nanmean([r['E']['ext_inland_med'] for r in rs])
            print(f"      extended inland (0.05-0.15 at 29-33 km): {sh:.3f} at {md:.1f} km {'ok' if .05 <= sh <= .15 and 29 <= md <= 33 else 'x'}")
        led = [json.loads((Path(sd)/'stats.json').read_text()).get('present', {}) for sd in seeds_of[name]]
        if led and all('area_ledger_frac' in x for x in led):
            keys = sorted({k for x in led for k in x['area_ledger_frac']})
            net = np.mean([x['continental_area_change_frac'] for x in led])
            parts = {k: np.mean([x['area_ledger_frac'].get(k, 0.) for x in led]) for k in keys}
            ends = [e for x in led for e in x.get('extension_ends', [])]
            print(f"      area ledger (share of the surface, 250 Myr): net {net:+.3f} = " + ' '.join(f'{k} {v:+.3f}' for k, v in parts.items())
                  + f" + residual {net-sum(parts.values()):+.3f}" + (f"; E events ended {len(ends)/len(led):.1f} per run, median "
                  f"{np.median([e['myr'] for e in ends]):.0f} Myr, guard-ended {sum(e['guard'] for e in ends)}" if ends else '')
                  + (f", lost (a half merged away) {np.mean([x.get('extension_lost', 0) for x in led]):.1f} per run"
                     if any('extension_lost' in x for x in led) else ''))
        trig = [x['extension_trigger'] for x in led if x.get('extension_trigger', {}).get('gt_onset_steps_mkm2', 0) > 0]
        if trig:                                     # E2 (twenty-ninth pass): design check, not a pass band
            m_ = {k: np.mean([t_[k] for t_ in trig]) for k in ('eligible', 'hot_pieces', 'held_pieces', 'hot_nodes')}
            sts = [e for x in led for e in x.get('extension_starts', [])]
            print(f"      E trigger (share of > onset area over the run): eligible {m_['eligible']:.2f} (design check >= 0.5 "
                  f"{'ok' if m_['eligible'] >= .5 else 'x'}), in hot pieces {m_['hot_pieces']:.2f}, in held pieces "
                  f"{m_['held_pieces']:.2f}, hot nodes {m_['hot_nodes']:.2f}"
                  + (f"; starts {len(sts)/len(led):.1f} per run, across (held) {sum(bool(e.get('across')) for e in sts)}, "
                     f"median hot share {np.median([e.get('hot_share', np.nan) for e in sts]):.2f}" if sts else '')
                  + (f"; band cuts {sum(e.get('cut') == 'band' for e in sts)}, median band half-length "
                     f"{np.nanmedian([e.get('L') or np.nan for e in sts])/1e3:.0f} km, far side on a collision (node share, "
                     f"median) {np.nanmedian([e.get('far_collision', np.nan) for e in sts]):.2f}"
                     if any(e.get('cut') == 'band' for e in sts) else ''))
            if sts and any(e.get('cut') == 'band' for e in sts):     # E4 design checks (thirty-second pass)
                lost = np.mean([x.get('extension_lost', 0) for x in led]); nst = len(sts)/len(led)
                fcm = float(np.nanmedian([e.get('far_collision', np.nan) for e in sts]))
                fs = [e for e in sts if e.get('side') is not None]
                print(f"      E4 design checks: lost {lost/max(nst, 1e-9):.2f} of starts (<= 0.33 {'ok' if lost <= nst/3 else 'x'}), "
                      f"far side on a collision median {fcm:.2f} (<= 0.2 {'ok' if fcm <= .2 else 'x'})"
                      + (f"; side rule {fs[0]['side']}, larger side moved in {sum(bool(e.get('moved_larger')) for e in fs)} "
                         f"of {len(fs)} band starts" if fs else ''))
        eok = [abs(np.log10(E['slope_p50']/EARTH_E['slope_p50'])) <= .2, E['slope_lt5'] <= .35,
               abs(np.log10(E['depth_p50']/EARTH_E['depth_p50'])) <= .2, abs(np.log10(E['depth_p90']/EARTH_E['depth_p90'])) <= .2]
        print(f"      E tests (proposed 25 Oct, not counted above): passes {sum(eok)} of 4 | slope > 2 km p50 {E['slope_p50']:.1f} m/km "
              f"(7.0-17.6) {'ok' if eok[0] else 'x'}; under 5 m/km {E['slope_lt5']:.2f} (<= 0.35) {'ok' if eok[1] else 'x'}; > 45 km "
              f"depth p50 {E['depth_p50']:.0f} km (58-146) {'ok' if eok[2] else 'x'}, p90 {E['depth_p90']:.0f} km (203-510) {'ok' if eok[3] else 'x'}")


if __name__ == '__main__':
    main()
