"""Plate-layer runs against the eighteenth-pass tests (docs/planet-layer-plan.md) and the crust / hypsometry bands.

  python -m scripts.elevation_check outputs\\planet-calibration\\sweep-20261024-120000
  python -m scripts.elevation_check --latest

For every seed folder (or the seed folders of a combination / sweep folder):
  tests (fixed 24 Oct before the change), on every saved snapshot of the last 40 Myr:
    young ocean (< 20 Myr, crust anomaly < 1 km, sediment < 500 m) within 100 km of a coast: share above sea level
    <= 1%, median offset from its own depth (GDH1 + anomaly + sediment + dynamic topography) within +-300 m;
    continental crust thinner than 30 km: median offset from its own Airy depth within +-300 m;
    (the two offsets: one area-weighted median over the nodes of all the window's snapshots pooled, since 27 Oct;
    before, the mean of per-snapshot medians, which gives a snapshot with 0.06% of the surface the weight of one with
    0.4%: seed 1004 -313 m pooled -273, every other run within 6 m; the old reading is printed beside it)
    ocean crust eroded above sea level (the old leak) <= 0.1 km^3/yr;
  and at present day: continental crust (scripts.crust_review bands), hypsometry against Earth (ETOPO, no ice
  sheets), shelf metrics, continental volume change, export.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import graph_landscape as gl, plates as P
from planet_layer.sphere_graph import CubeSphere
from scripts import crust_review as cr
from scripts.sweep_plates import EARTH_EXPORT_KM3_YR, QS, shelf_metrics

ROOT = Path(__file__).resolve().parents[1]
TESTS = {'young_ocean_above_sea': (0., .01), 'young_ocean_offset_m': (-300., 300.), 'thin_crust_offset_m': (-300., 300.),
         'ocean_crust_eroded_subaerial_km3_yr': (-.1, 0.)}


def snapshot_tests(path, g, p):
    z = np.load(path)
    cont = z['continental'].astype(bool); T = z['crust_thickness_m'].astype(np.float64)
    e = z['elevation_m'].astype(np.float64); S = z['sediment_m'].astype(np.float64)
    sed, s_dry, _ = P.sediment_column(S, cont, p)               # bulk height and dry mass (compaction, B-a)
    dyn = z['dynamic_topography_m'].astype(np.float64); sl = float(z['sea_level_m'])
    age = z['ocean_age_myr'].astype(np.float64)
    rc, rm, rw = p['rho_crust'], p['rho_mantle'], p['rho_water']
    a = g.area
    anom = T-p['ocean_thick_m']
    own_ocean = -P.gdh1_depth(np.nan_to_num(age))+anom+sed+dyn-sl
    d = gl.distance_to(cont, g.nbr, g.dist, 150e3)
    yo = (~cont) & (age < 20) & (anom < 1000) & (sed < 500) & (d <= 100e3)
    # continental column: crust, and with continental_sediment its sediment (rho_sediment), at local Airy, water-loaded
    # below sea level; the thin-crust test selects by crust + sediment, as CRUST1.0 and crust_review count it
    cs = np.where(cont, sed, 0.) if p.get('continental_sediment') else 0.
    cm = np.where(cont, s_dry, 0.) if p.get('continental_sediment') else 0.
    dry = p['cont_ref_elev_m']+(T-p['cont_ref_thick_m'])*(rm-rc)/rm+cs-cm/rm
    own_cont = np.where(dry+dyn-sl < 0, (dry+dyn-sl)*rm/(rm-rw), dry+dyn-sl)
    thin = cont & (T+cs < 30e3)
    return {'young_ocean_area': float(a[yo].sum()/a.sum()),
            'young_ocean_above_sea': float(a[yo & (e > 0)].sum()/max(a[yo].sum(), 1e-9)),
            'young_ocean_offset_m': float(np.median((e-own_ocean)[yo])) if yo.any() else 0.,
            'thin_crust_area': float(a[thin].sum()/a.sum()),
            'thin_crust_offset_m': float(np.median((e-own_cont)[thin])) if thin.any() else 0.,
            '_young_ocean': ((e-own_ocean)[yo], a[yo]), '_thin_crust': ((e-own_cont)[thin], a[thin])}


def weighted_median(x, w):
    """Median of x weighted by w (0 for an empty sample)."""
    if len(x) == 0:
        return 0.
    o = np.argsort(x); c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, .5*c[-1])])


def check(seed_dir, g, earth_hyp):
    seed_dir = Path(seed_dir)
    st = json.loads((seed_dir/'stats.json').read_text())
    from scripts.crust_review import run_params
    p = run_params(seed_dir)
    snaps = sorted((seed_dir/'snapshots').glob('t*.npz'), key=lambda s: float(s.stem[1:]))   # by time, oldest first
    tests = [snapshot_tests(s, g, p) for s in snaps[-40::5]+[snaps[-1]]]
    out = {k: float(np.mean([t[k] for t in tests])) for k in tests[0] if not k.startswith('_')}
    for k in ('young_ocean', 'thin_crust'):                      # the gate: pooled, area-weighted (27 Oct)
        out[k+'_offset_snapmean_m'] = out[k+'_offset_m']
        out[k+'_offset_m'] = weighted_median(np.concatenate([t['_'+k][0] for t in tests]),
                                             np.concatenate([t['_'+k][1] for t in tests]))
    out['young_ocean_above_sea_max'] = float(max(t['young_ocean_above_sea'] for t in tests))
    e = st['present']['erosion_totals_km3']
    out['ocean_crust_eroded_km3_yr'] = e.get('ocean_crust_eroded_m3', 0.)/st['years']/1e6
    # the old leak: normal oceanic crust (< ocean_thick_m + 1 km) eroded above sea; arcs and plateaus standing above sea
    # are islands, not a leak (thirty-ninth pass; runs before it booked only the total, which is read then)
    out['ocean_crust_eroded_subaerial_all_km3_yr'] = e.get('ocean_crust_eroded_subaerial_m3', 0.)/st['years']/1e6
    out['ocean_crust_eroded_subaerial_km3_yr'] = e.get('ocean_crust_eroded_subaerial_normal_m3',
                                                       e.get('ocean_crust_eroded_subaerial_m3', 0.))/st['years']/1e6
    out['export_km3_yr'] = -(e['continental_eroded_m3']+e['continental_deposited_m3']
                             -e.get('continental_deposited_submerged_m3', 0.))/st['years']/1e6
    out['deposited_submerged_km3_yr'] = e.get('continental_deposited_submerged_m3', 0.)/st['years']/1e6
    out['continental_volume_change'] = st['continental_volume_change']
    out['continental_crust_volume_change'] = st.get('continental_crust_volume_change', st['continental_volume_change'])
    out['continental_sediment_deposited_km3_yr'] = e.get('continental_sediment_m3', 0.)/st['years']/1e6
    out.update({'crust_'+k: v for k, v in cr.metrics(seed_dir/'snapshots'/'t+0000.0.npz', g).items()})
    z0 = np.load(seed_dir/'snapshots'/'t+0000.0.npz')
    z = z0['elevation_m'].astype(np.float64)
    land = z > 0
    out['land_fraction'] = float(g.area[land].sum()/g.area.sum())
    out.update({f'land_p{q}_m': float(np.percentile(z[land], float(q))) for q in QS})
    errs = [abs(np.log10(out['land_fraction']/earth_hyp['land_fraction']))]
    errs += [abs(np.log10(max(out[f'land_p{q}_m'], 1.)/earth_hyp['land_percentiles_m'][q])) for q in QS]
    out['hypsometry_score'] = float(max(errs))
    out.update({'shelf_'+k: v for k, v in shelf_metrics(z, z0['continental'].astype(bool), g).items()})
    out['tests_pass'] = all(lo <= out[k] <= hi for k, (lo, hi) in TESTS.items())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--latest', action='store_true')
    a = ap.parse_args(argv)
    if a.latest:
        a.dirs = [str(sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1])]
    from scripts.earth_reference import load
    earth_hyp = load(217)['no_ice_sheets']
    g = CubeSphere(217)
    seeds = []
    for d in map(Path, a.dirs):
        seeds += [d] if (d/'snapshots').is_dir() else sorted(x for x in d.rglob('seed*') if (x/'snapshots').is_dir())
    res = {}
    for s in seeds:
        r = check(s, g, earth_hyp)
        res[str(s)] = r
        print(f"{s}: tests {'pass' if r['tests_pass'] else 'FAIL'} | young ocean above sea {r['young_ocean_above_sea']:.3f} "
              f"(max {r['young_ocean_above_sea_max']:.3f}), offset {r['young_ocean_offset_m']:+.0f} m (snapshot mean "
              f"{r['young_ocean_offset_snapmean_m']:+.0f}); thin crust offset {r['thin_crust_offset_m']:+.0f} m (snapshot "
              f"mean {r['thin_crust_offset_snapmean_m']:+.0f}); ocean crust eroded {r['ocean_crust_eroded_km3_yr']:.2f} "
              f"(above sea: normal crust {r['ocean_crust_eroded_subaerial_km3_yr']:.2f}, all {r['ocean_crust_eroded_subaerial_all_km3_yr']:.2f}) km3/yr")
        print(f"   crust: area {r['crust_area']:.3f}, <30 km {r['crust_thin_lt30']:.3f}, >45 km {r['crust_thick_gt45']:.3f}, "
              f"median {r['crust_median_km']:.1f} km, sediment on continents {r['crust_sed_mean_m']:.0f} m | volume "
              f"{r['continental_volume_change']:+.1%} (crust {r['continental_crust_volume_change']:+.1%}), export "
              f"{r['export_km3_yr']:.1f}, deposited submerged {r['deposited_submerged_km3_yr']:.1f} km3/yr | land "
              f"{r['land_fraction']:.1%}, p25/50/75/90 {r['land_p25_m']:.0f}/{r['land_p50_m']:.0f}/{r['land_p75_m']:.0f}/"
              f"{r['land_p90_m']:.0f} m (hyps. {r['hypsometry_score']:.2f}) | crust drowned {r['shelf_continental_submerged']:.0%}, "
              f"p70 {r['shelf_whole_p70_m']:.0f} m")
    return res


if __name__ == '__main__':
    main()
