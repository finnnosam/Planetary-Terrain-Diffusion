"""Continental budget and ocean basin of plate-layer runs against Earth (item B, plan doc thirty-fifth pass).

  python -m scripts.crust_budget outputs\\planet-calibration\\sweep-20261025-170555\\boundary_normal-ring_extension-False_ext_onset_thick_m-45000
  python -m scripts.crust_budget --latest

Reads only what the runs saved (stats.json series every 1 Myr, snapshots every 10 Myr before -40 Myr and every 1 Myr
after); changes nothing in the model. For every combination folder (or seed folder):

1. B's integrated tests (fixed 26 Oct, thirty-fourth pass; Earth computed here from ETOPO on the same grid where it
   can be): water the basins hold below the datum (global equivalent layer) within +-100 m of Earth's; continental
   volume change over the run >= -3%; label-free continental area 0.384 +- 0.03; deep-ocean floor mean (deeper than
   1,500 m below the datum) within +-150 m of Earth's; sediment subduction <= 1.7 km^3/yr.
2. The budget in km^3/yr over the whole run and over the last 40 Myr (Earth's fluxes are present-day rates): the
   continental crust (crystalline), the sediment on it, the ocean floor's sediment, and island arcs (oceanic crust
   thicker than ocean_thick_m), each with its closure residual.
3. Island arcs: the stock and its thickness, where it sits, what reaches trenches (last 40 Myr, 1 Myr snapshots:
   arc and plateau crust on the downgoing side of a subduction boundary, times the closing rate, facing continental
   or oceanic crust), and the loss by difference (growth - change of stock - turned continental + eroded).
4. Trench length (contour, scripts.boundary_lengths; continental vs intra-oceanic) and per-km rates.

Earth (Scholl & von Huene 2009, Geol. Soc. Spec. Publ. 318; 1 AU = 1 km^3/yr): arc addition 2.7 (intra-oceanic 1.5,
continental margins 1.0, collisional 0.2); losses 3.2: ocean-margin subduction zones 2.5 (~60 km^3/Myr per km of
trench, ~60% subduction erosion, ~40% sediment subduction), collision zones 0.7 (~115 km^3/Myr/km over ~6,000 km);
net growth effectively nil. Some terms are booked net in the plate layer (arc addition and subduction erosion on the
same node, plates._subduction_thickening); they are printed as booked and say so.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer.sphere_graph import CubeSphere
from scripts.crust_review import column_thickness_m, label_free_continent, run_params

ROOT = Path(__file__).resolve().parents[1]
EARTH_N217 = ROOT/'outputs'/'planet-reference'/'earth-n217.npz'
YR = 1e6                                     # series and snapshots are 1 Myr apart; rates per year
# Scholl & von Huene 2009 (km^3/yr; km^3/Myr per km of trench)
EARTH_BUDGET = {'arc_total': 2.7, 'arc_intra_oceanic': 1.5, 'arc_continental_margin': 1.0, 'arc_collisional': .2,
                'loss_total': 3.2, 'loss_ocean_margin': 2.5, 'loss_collision': .7,
                'ocean_margin_per_km': 60., 'subduction_erosion_share': .6, 'collision_per_km': 115.}
EARTH_SEDIMENT_SUBDUCTED = 1.0               # 40% of 2.5 (the 2009 split); older estimates 1.65 (von Huene & Scholl 1991)
EARTH_AREA = .384                            # label-free (CRUST1.0 total >= 20 km), crust_review
TESTS = {'water_below_datum_m': 100., 'volume_change_min': -.03, 'area': .03, 'floor_mean_m': 150.,
         'sediment_subducted_max': 1.7}
# island-arc tests (B-d, fixed 27 Oct before its runs; docs/planet-layer-plan.md, thirty-ninth pass):
#  A1 arc growth net of forearc erosion per km of intra-oceanic trench, last 40 Myr, 30-95 km^3/Myr/km: Dimalanta
#     et al. 2002 (EPSL 202), western Pacific arc crust volume / age since subduction began, crust 20-30 km;
#  A2 arcs mature: the arc stock's thickness p90 >= 15 km at present (arcs leave the stock at 20 km, when they turn
#     continental; Earth's island arcs 26 +- 6 km, Tetreault & Buiter 2014), and arc crust reaching the continents
#     over the run (turned continental + docked with arc_accretion) > 0 on every seed;
#  A3 (design check) arc crust lost by difference (subducted) <= half of the run's arc growth: on Earth arcs are an
#     addition to the continents (Scholl & von Huene 2009: intra-oceanic 1.5 km^3/yr), not a stock the trenches recycle
ARC_TESTS = {'growth_per_km': (30., 95.), 'stock_p90_km_min': 15., 'lost_share_max': .5}
FLOOR_BELOW_M = 1500.


def earth_ocean(g):
    """Earth on the n = 217 graph (ETOPO 2022 surface, scripts.earth_reference): water below sea level (GEL, m), the
    mean depth and area share of the floor deeper than FLOOR_BELOW_M."""
    z = np.load(EARTH_N217)['elevation_m'].astype(np.float64)
    a = g.area
    deep = z < -FLOOR_BELOW_M
    return {'water_below_datum_m': float((np.clip(-z, 0, None)*a).sum()/a.sum()),
            'floor_mean_m': float(-np.average(z[deep], weights=a[deep])), 'floor_share': float(a[deep].sum()/a.sum())}


def seed_dirs(dirs):
    combos = []
    for d in map(Path, dirs):
        if (d/'snapshots').is_dir():
            combos.append((d.parent.name, [d]))
        elif any(d.glob('seed*/snapshots')):
            combos.append((d.name, sorted(d.glob('seed*'))))
        else:
            combos += [(c.name, sorted(c.glob('seed*'))) for c in sorted(d.iterdir())
                       if c.is_dir() and any(c.glob('seed*/snapshots'))]
    return combos


def snap(seed, t):
    return np.load(Path(seed)/'snapshots'/f't{t:+07.1f}.npz')


def ocean_tests(z, g, p):
    """Water below the datum, sea level, deep floor, label-free area, from one snapshot."""
    a = g.area
    s = float(z['sea_level_m'])
    zd = z['elevation_m'].astype(np.float64)+s                       # elevation on the datum
    deep = zd < -FLOOR_BELOW_M
    cont = z['continental'].astype(bool)
    Tcol = column_thickness_m(z, p)
    return {'sea_level_on_datum_m': s,
            'water_below_datum_m': float((np.clip(-zd, 0, None)*a).sum()/a.sum()),
            'water_check_m': float((np.clip(s-zd, 0, None)*a).sum()/a.sum()),   # the inventory, as a check
            'floor_mean_m': float(-np.average(zd[deep], weights=a[deep])), 'floor_share': float(a[deep].sum()/a.sum()),
            'oceanic_flag_share': float(a[~cont].sum()/a.sum()),
            'area': float(a[label_free_continent(Tcol)].sum()/a.sum())}


def arc_stock(z, g, p, near_d=None):
    """Oceanic crust thicker than ocean_thick_m (island arcs, and hotspot / LIP crust on the ocean floor): volume (km^3),
    thickness of the nodes carrying >= 1 km of it, and the share within 300 km of a subduction boundary."""
    a = g.area
    cont = z['continental'].astype(bool)
    T = z['crust_thickness_m'].astype(np.float64)
    an = np.where(cont, 0., np.clip(T-p['ocean_thick_m'], 0, None))
    out = {'stock_km3': float((an*a).sum()/1e9)}
    m = an >= 1e3
    out['area_ge1km_share_of_ocean'] = float(a[m].sum()/a[~cont].sum())
    if m.any():
        o = np.argsort(T[m]); c = np.cumsum(a[m][o]); c /= c[-1]
        out['thick_p50_km'] = float(T[m][o][np.searchsorted(c, .5)]/1e3)
        out['thick_p90_km'] = float(T[m][o][np.searchsorted(c, .9)]/1e3)
        out['thick_max_km'] = float(T[m].max()/1e3)
    if near_d is not None:
        out['share_within_300km_of_subduction'] = float((an*a)[near_d < 300e3].sum()/max((an*a).sum(), 1.))
    return out


def subduction_distance(z, g):
    from planet_layer.plates import SUBDUCTION, nearest_label
    lab = np.where(z['boundary_type'] == SUBDUCTION, np.arange(g.size), -1).astype(np.int64)
    if (lab < 0).all():
        return np.full(g.size, np.inf)
    _, d = nearest_label(lab, g.nbr, g.dist)
    return d


def trench_arrivals(z, g, p):
    """Arc / plateau crust and sediment on the downgoing side of subduction boundaries, carried in at the closing rate
    (km^3/yr), split by what the other side is (continental crust: a terrane would dock on Earth; oceanic). The
    downgoing side: oceanic against continental; of two oceanic sides the older (the plate layer keeps one polarity
    per plate pair from the mean contact age, so this is a node-wise approximation). Node lengths: each downgoing
    boundary node stands for one cell width; the node count overstates line length (scripts.boundary_lengths: 1.2-
    1.4x), so run_one scales the result by contour / node length of the subduction boundary."""
    from planet_layer.plates import SUBDUCTION
    pid = z['plate_id']; cont = z['continental'].astype(bool)
    age = z['crust_age_myr'].astype(np.float64)
    bt = z['boundary_type']
    idx = np.flatnonzero((bt == SUBDUCTION) & ~cont)
    v = np.clip(z['convergence_mm_yr'].astype(np.float64)[idx], 0, None)*1e-3   # m/yr
    nbi = g.nbr[idx]
    ok = nbi >= 0
    nbs = np.where(ok, nbi, 0)
    other = ok & (pid[nbs] != pid[idx][:, None])
    faces_cont = (other & cont[nbs]).any(1)
    oage = np.where(other & ~cont[nbs], age[nbs], np.nan)
    has = np.isfinite(oage).any(1)
    mean_o = np.where(has, np.nansum(np.where(np.isfinite(oage), oage, 0.), 1)/np.maximum(np.isfinite(oage).sum(1), 1), np.inf)
    older = age[idx] >= mean_o
    down = faces_cont | older
    h = np.sqrt(g.area.mean())
    an = np.clip(z['crust_thickness_m'].astype(np.float64)[idx]-p['ocean_thick_m'], 0, None)
    sed = z['sediment_m'].astype(np.float64)[idx]
    out = {}
    for name, m in (('facing_continent', down & faces_cont), ('facing_ocean', down & ~faces_cont)):
        out[f'arc_{name}_km3_yr'] = float((an*v*h)[m].sum()/1e9)
        out[f'sed_{name}_km3_yr'] = float((sed*v*h)[m].sum()/1e9)
        out[f'length_node_{name}_km'] = float(m.sum()*h/1e3)
    return out


def contour_lengths(seed, t, g):
    from scripts.boundary_lengths import lengths
    z = snap(seed, t)
    pj = Path(seed)/'snapshots'/f't{t:+07.1f}-plates.json'
    return lengths(g, z['plate_id'], z['boundary_type'], plates=json.loads(pj.read_text()), continental=z['continental'])


def series_rates(stats, t0, t1):
    """km^3/yr between t0 and t1 for every ledger in the series (the series stores km^3, whatever the key says)."""
    S = {round(r['time_myr'], 3): r for r in stats['series']}
    a, b = S[round(t0, 3)], S[round(t1, 3)]
    dt = (t1-t0)*YR
    out = {}
    for grp in ('erosion_totals_km3', 'sediment_ledger_km3', 'area_ledger_frac'):
        for k in set(a.get(grp, {})) | set(b.get(grp, {})):
            out[f'{grp}.{k}'] = (b.get(grp, {}).get(k, 0.)-a.get(grp, {}).get(k, 0.))/dt
    out['crust_net'] = (b['continental_volume_km3']-a['continental_volume_km3'])/dt
    out['sediment_net'] = (b['continental_sediment_km3']-a['continental_sediment_km3'])/dt
    return out


def budget(seed, g, t0, t1, p, stats):
    """The budget of one run between t0 and t1 (Myr), km^3/yr."""
    r = series_rates(stats, t0, t1)
    E = lambda k: r.get('erosion_totals_km3.'+k, 0.)
    A = 4*np.pi*(g.radius/1e3)**2                                     # km^2
    arc_conv = r.get('area_ledger_frac.arc', 0.)*A*p['arc_to_continent_thick_m']/1e3   # km^3/yr, lower bound
    crust_terms = {'crystalline_erosion': E('continental_crust_eroded_m3'),
                   'trench_on_continents_net': E('subduction_on_continents_m3'),
                   'sediment_accreted_at_trenches': E('sediment_accreted_m3'),
                   'collision_sediment_to_crust': E('collision_sediment_to_crust_m3'),
                   'volcanic_all_nodes': E('volcanic_added_m3'),
                   'arcs_turned_continental': arc_conv,
                   'arcs_docked_on_continents': E('arc_accreted_on_continent_m3')}
    za, zb = snap(seed, t0), snap(seed, t1)
    ca, cb = (z['continental'].astype(bool) for z in (za, zb))
    sa, sb = ((z['sediment_m'].astype(np.float64)*g.area)[~c].sum()/1e9 for z, c in ((za, ca), (zb, cb)))
    da, db = subduction_distance(za, g), subduction_distance(zb, g)
    arc_a, arc_b = arc_stock(za, g, p, da), arc_stock(zb, g, p, db)
    growth = E('subduction_on_ocean_arcs_m3')
    d_stock = (arc_b['stock_km3']-arc_a['stock_km3'])/((t1-t0)*YR)
    eroded = E('ocean_crust_eroded_m3')                              # <= 0
    out = {'window_myr': [t0, t1],
           'crust': {'net': r['crust_net'], **crust_terms, 'residual': r['crust_net']-sum(crust_terms.values())},
           'sediment_on_continents': {'net': r['sediment_net'], 'eroded_total': E('continental_eroded_m3'),
                                      'deposited': E('continental_deposited_m3'),
                                      'deposited_submerged': E('continental_deposited_submerged_m3'),
                                      'shelf_off_land': E('shelf_off_land_m3')},
           'ocean_floor_sediment': {'from_continents': E('ocean_sediment_m3'), 'of_it_deep_sea_fill': E('deep_sea_sediment_m3'),
                                    'subducted': E('sediment_subducted_m3'), 'accreted': E('sediment_accreted_m3'),
                                    'stock_change': (sb-sa)/((t1-t0)*YR), 'stock_end_km3': sb},
           'arcs': {'growth_net_of_forearc_erosion': growth, 'stock_change': d_stock, 'turned_continental': arc_conv,
                    'eroded': eroded, 'docked_on_continents': E('arc_accreted_on_continent_m3'),
                    'lost_by_difference': growth+eroded-d_stock-arc_conv-E('arc_accreted_on_continent_m3'),
                    'volcanic_all_nodes_not_in_growth': E('volcanic_added_m3'),
                    # trench_flux 'consumed' (B-d2): consumed oceanic area routed to an overriding trench / not (ratio only)
                    'flux_routed': E('trench_flux_routed_m2'), 'flux_unrouted': E('trench_flux_unrouted_m2'),
                    'stock_start': arc_a, 'stock_end': arc_b},
           'area_ledger_per_myr': {k.split('.', 1)[1]: v*YR for k, v in r.items() if k.startswith('area_ledger_frac.')}}
    os_ = out['ocean_floor_sediment']
    os_['residual'] = os_['stock_change']-(os_['from_continents']-os_['subducted']-os_['accreted'])
    return out


def run_one(seed, g, earth):
    stats = json.loads((Path(seed)/'stats.json').read_text())
    p = run_params(seed)
    T0 = stats['series'][0]['time_myr']
    tests_start = ocean_tests(snap(seed, T0), g, p)
    tests = ocean_tests(snap(seed, 0.), g, p)
    tests['volume_change'] = stats['continental_volume_change']
    tests['crust_volume_change'] = stats['continental_crust_volume_change']
    whole = budget(seed, g, T0, 0., p, stats)
    recent = budget(seed, g, -40., 0., p, stats)
    arr = [trench_arrivals(snap(seed, float(t)), g, p) for t in range(-40, 1)]
    arrivals = {k: float(np.mean([x[k] for x in arr])) for k in arr[0]}
    L = [contour_lengths(seed, float(t), g) for t in (-40, -30, -20, -10, 0)]
    lens = {k: float(np.mean([x.get(k, 0.) for x in L])) for k in ('subduction_km', 'subduction_continental_km',
                                                                  'subduction_oceanic_km', 'collision_km')}
    node_down = arrivals['length_node_facing_continent_km']+arrivals['length_node_facing_ocean_km']
    scale = lens['subduction_km']/node_down if node_down > 0 else 1.
    for k in list(arrivals):
        if k.endswith('_km3_yr'):
            arrivals[k+'_contour'] = arrivals[k]*scale
    arrivals['contour_over_node_length'] = scale
    km = lens['subduction_km']
    os_ = recent['ocean_floor_sediment']
    per_km = {'sediment_subducted': os_['subducted']*1e6/km,
              'sediment_subducted_plus_accreted': (os_['subducted']+os_['accreted'])*1e6/km,
              'trench_on_continents_net_per_cont_km': recent['crust']['trench_on_continents_net']*1e6/max(lens['subduction_continental_km'], 1.),
              'arc_growth_net_per_oceanic_km': recent['arcs']['growth_net_of_forearc_erosion']*1e6/max(lens['subduction_oceanic_km'], 1.)}
    return {'tests': tests, 'tests_start': tests_start, 'whole': whole, 'recent': recent, 'arrivals': arrivals,
            'lengths_km_last40': lens, 'per_km_km3_myr': per_km}


def mean_tree(xs):
    if isinstance(xs[0], dict):
        return {k: mean_tree([x[k] for x in xs]) for k in xs[0] if all(k in x for x in xs)}
    if isinstance(xs[0], (int, float)) and not isinstance(xs[0], bool):
        return float(np.mean(xs))
    return xs[0]


def passes(m, earth):
    t = m['tests']
    return {'water': abs(t['water_below_datum_m']-earth['water_below_datum_m']) <= TESTS['water_below_datum_m'],
            'volume': t['volume_change'] >= TESTS['volume_change_min'],
            'area': abs(t['area']-EARTH_AREA) <= TESTS['area'],
            'floor': abs(t['floor_mean_m']-earth['floor_mean_m']) <= TESTS['floor_mean_m'],
            'sediment_subducted': m['recent']['ocean_floor_sediment']['subducted'] <= TESTS['sediment_subducted_max']}


def arc_tests(m, res):
    """B-d's island-arc tests (ARC_TESTS) on a combination's seed means; per-seed values for A2."""
    g = m['per_km_km3_myr']['arc_growth_net_per_oceanic_km']
    p90 = m['whole']['arcs']['stock_end'].get('thick_p90_km', 0.)
    conv = [r['whole']['arcs']['turned_continental']+r['whole']['arcs'].get('docked_on_continents', 0.) for r in res]
    wa = m['whole']['arcs']
    lost = wa['lost_by_difference']/max(wa['growth_net_of_forearc_erosion'], 1e-12)
    lo, hi = ARC_TESTS['growth_per_km']
    return {'A1': lo <= g <= hi, 'A2': p90 >= ARC_TESTS['stock_p90_km_min'] and min(conv) > 0, 'A3': lost <= ARC_TESTS['lost_share_max'],
            'growth_per_km': g, 'stock_p90_km': p90, 'turned_continental_by_seed': conv, 'lost_share': lost}


def report(name, res, earth):
    m = mean_tree(res)
    t, ts = m['tests'], m['tests_start']
    ok = passes(m, earth)
    P = lambda k: 'pass' if ok[k] else 'FAIL'
    f = lambda x: f'{x:+.2f}'
    by_seed = lambda fn: ' / '.join(f'{fn(r):.0f}' for r in res)
    print(f"\n=== {name} ({len(res)} seeds) ===")
    print("B tests (means at present; at the start in brackets):")
    print(f"  water below the datum {t['water_below_datum_m']:.0f} m ({ts['water_below_datum_m']:.0f}; seeds "
          f"{by_seed(lambda r: r['tests']['water_below_datum_m'])}) vs Earth {earth['water_below_datum_m']:.0f} +-100: {P('water')}; "
          f"sea level on the datum {t['sea_level_on_datum_m']:.0f} m ({ts['sea_level_on_datum_m']:.0f}); water inventory "
          f"{t['water_check_m']:.0f} m")
    print(f"  deep floor (below -1.5 km on the datum) mean {t['floor_mean_m']:.0f} m on {t['floor_share']:.3f} of the surface "
          f"({ts['floor_mean_m']:.0f} on {ts['floor_share']:.3f}) vs Earth {earth['floor_mean_m']:.0f} on {earth['floor_share']:.3f} "
          f"+-150: {P('floor')}; oceanic flag {t['oceanic_flag_share']:.3f}")
    print(f"  continental volume change {t['volume_change']:+.1%} (crust {t['crust_volume_change']:+.1%}) >= -3%: {P('volume')}; "
          f"label-free area {t['area']:.3f} ({ts['area']:.3f}) vs {EARTH_AREA} +-0.03: {P('area')}")
    for key, lab in (('whole', 'whole run'), ('recent', 'last 40 Myr')):
        b = m[key]
        c, s, o, a = b['crust'], b['sediment_on_continents'], b['ocean_floor_sediment'], b['arcs']
        print(f"Budget, {lab} (km^3/yr):")
        print(f"  continental crust net {f(c['net'])} = crystalline erosion {f(c['crystalline_erosion'])} + trench on continents "
              f"(arc - subduction erosion, booked net) {f(c['trench_on_continents_net'])} + sediment accreted at trenches "
              f"{f(c['sediment_accreted_at_trenches'])} + collision sediment {f(c['collision_sediment_to_crust'])} + volcanic "
              f"(all nodes) {f(c['volcanic_all_nodes'])} + arcs turned continental {f(c['arcs_turned_continental'])} + arcs docked "
              f"{f(c['arcs_docked_on_continents'])} + residual "
              f"{f(c['residual'])}")
        print(f"  continental surface: eroded {f(s['eroded_total'])}, deposited {f(s['deposited'])} (submerged "
              f"{f(s['deposited_submerged'])}), sediment on continents net {f(s['net'])}")
        print(f"  ocean-floor sediment: from continents {f(o['from_continents'])} (deep-sea fill {f(o['of_it_deep_sea_fill'])}), "
              f"subducted {f(-o['subducted'])}, accreted {f(-o['accreted'])}, stock change {f(o['stock_change'])}, residual "
              f"{f(o['residual'])}; stock {o['stock_end_km3']/1e6:.0f} Mkm^3")
        print(f"  island arcs: growth (net of forearc erosion) {f(a['growth_net_of_forearc_erosion'])}, stock change "
              f"{f(a['stock_change'])}, turned continental {f(a['turned_continental'])}, docked on continents "
              f"{f(a['docked_on_continents'])}, eroded {f(a['eroded'])}, lost by "
              f"difference {f(a['lost_by_difference'])} (volcanic on all nodes, not in growth: {f(a['volcanic_all_nodes_not_in_growth'])})")
        print("  area ledger per Myr (share of the surface): "+', '.join(f'{k} {v:+.5f}' for k, v in sorted(b['area_ledger_per_myr'].items())))
    a0, a1 = m['whole']['arcs']['stock_start'], m['whole']['arcs']['stock_end']
    print(f"Island-arc stock: start {a0['stock_km3']/1e6:.1f} Mkm^3, present {a1['stock_km3']/1e6:.1f} Mkm^3; nodes with >= 1 km: "
          f"{a1['area_ge1km_share_of_ocean']:.3f} of the ocean floor, total thickness p50 / p90 / max {a1.get('thick_p50_km', 0):.1f} / "
          f"{a1.get('thick_p90_km', 0):.1f} / {a1.get('thick_max_km', 0):.1f} km (turns continental at 20 km); within 300 km of a "
          f"subduction boundary {a1.get('share_within_300km_of_subduction', 0):.2f}")
    r = m['arrivals']
    print(f"At trenches, last 40 Myr (downgoing side, contour-scaled x{r['contour_over_node_length']:.2f}): arc crust facing "
          f"continents {r['arc_facing_continent_km3_yr_contour']:.2f}, facing ocean {r['arc_facing_ocean_km3_yr_contour']:.2f} km^3/yr; "
          f"sediment facing continents {r['sed_facing_continent_km3_yr_contour']:.2f}, facing ocean {r['sed_facing_ocean_km3_yr_contour']:.2f} "
          f"(ledger subducted + accreted {m['recent']['ocean_floor_sediment']['subducted']+m['recent']['ocean_floor_sediment']['accreted']:.2f})")
    L = m['lengths_km_last40']; pk = m['per_km_km3_myr']
    print(f"Trench length (contour, mean of -40..0 every 10 Myr): subduction {L['subduction_km']/1e3:.1f}k km (continental side "
          f"{L['subduction_continental_km']/1e3:.1f}k, intra-oceanic {L['subduction_oceanic_km']/1e3:.1f}k), collision "
          f"{L['collision_km']/1e3:.1f}k; Bird 2003 subduction 51.3k")
    print(f"Per km of trench (km^3/Myr/km, last 40 Myr): sediment subducted {pk['sediment_subducted']:.1f} (+ accreted "
          f"{pk['sediment_subducted_plus_accreted']:.1f}) vs Earth ~{EARTH_BUDGET['ocean_margin_per_km']*(1-EARTH_BUDGET['subduction_erosion_share']):.0f}; "
          f"trench net on continents {pk['trench_on_continents_net_per_cont_km']:+.1f} per continental-side km; arc growth (net) "
          f"{pk['arc_growth_net_per_oceanic_km']:.1f} per intra-oceanic km")
    print(f"Earth (Scholl & von Huene 2009): arc addition {EARTH_BUDGET['arc_total']} (intra-oceanic {EARTH_BUDGET['arc_intra_oceanic']}, "
          f"continental margins {EARTH_BUDGET['arc_continental_margin']}, collisional {EARTH_BUDGET['arc_collisional']}); losses "
          f"{EARTH_BUDGET['loss_total']} (ocean margins {EARTH_BUDGET['loss_ocean_margin']}: ~{EARTH_BUDGET['ocean_margin_per_km']:.0f} "
          f"km^3/Myr/km, 60% subduction erosion; collisions {EARTH_BUDGET['loss_collision']}); sediment subducted ~{EARTH_SEDIMENT_SUBDUCTED}: "
          f"planet {m['recent']['ocean_floor_sediment']['subducted']:.2f} (<= 1.7: {P('sediment_subducted')})")
    at = arc_tests(m, res)
    Q = lambda k: 'pass' if at[k] else 'FAIL'
    print(f"Arc tests (B-d): A1 growth net of forearc erosion {at['growth_per_km']:.1f} km^3/Myr per intra-oceanic km "
          f"(30-95, Dimalanta et al. 2002): {Q('A1')}; A2 arc stock p90 {at['stock_p90_km']:.1f} km (>= 15) and turned "
          f"continental + docked by seed {' / '.join(f'{x:.3f}' for x in at['turned_continental_by_seed'])} km^3/yr (> 0): {Q('A2')}; "
          f"A3 lost by difference {at['lost_share']:.2f} of growth (<= 0.5): {Q('A3')}; beside: intra-oceanic addition "
          f"{m['recent']['arcs']['growth_net_of_forearc_erosion']:.2f} km^3/yr (Earth {EARTH_BUDGET['arc_intra_oceanic']}, on "
          f"{L['subduction_oceanic_km']/1e3:.1f}k km vs Earth ~20.8k)")
    wa = m['whole']['arcs']
    if wa.get('flux_routed', 0.) > 0:
        sh = wa['flux_routed']/(wa['flux_routed']+wa['flux_unrouted'])
        m['trench_flux_routed_share'] = sh
        print(f"Trench fluxes from consumed area (B-d2): routed share of the consumed oceanic area {sh:.3f} (design check >= 0.9: "
              f"{'pass' if sh >= .9 else 'FAIL'})")
    m['tests_pass'] = ok
    m['arc_tests'] = at
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*', help='a sweep folder, combination folders, or seed folders')
    ap.add_argument('--latest', action='store_true', help='the newest outputs/planet-calibration/sweep-* folder')
    ap.add_argument('--json', help='write the results here')
    a = ap.parse_args(argv)
    if a.latest:
        a.dirs = [str(sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1])]
        print('sweep', a.dirs[0])
    g = CubeSphere(217)
    earth = earth_ocean(g)
    print(f"Earth (ETOPO on n = 217): water below sea level {earth['water_below_datum_m']:.0f} m GEL; floor deeper than 1.5 km: "
          f"mean {earth['floor_mean_m']:.0f} m on {earth['floor_share']:.3f} of the surface")
    out = {'earth': earth, 'earth_budget': EARTH_BUDGET}
    for name, seeds in seed_dirs(a.dirs):
        res = [run_one(s, g, earth) for s in seeds]
        out[name] = {'mean': report(name, res, earth), 'seeds': {Path(s).name: r for s, r in zip(seeds, res)}}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == '__main__':
    main()
