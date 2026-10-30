"""Why docking island arcs (B-d) lengthens the trench and cuts plates: paired readings of the arc arms of a sweep
(docs/planet-layer-plan.md, thirty-ninth pass and after). No model change; reads stats.json and snapshots only.

  python -m scripts.arc_dock_probe outputs\\planet-calibration\\sweep-20261028-120000
  python -m scripts.arc_dock_probe --latest

Arms are the sweep's combination folders; the control is the one named by --control (default: the combination with
'arc_accretion-False' and 'arc_H_ocean_m-300' in its name). For every arm, seed by seed and as six-seed means:
 1. Timing (stats.json series, every 1 Myr): node trench length, ridge, plate count, mean speed; the first time the
    arm's trench stays >= 15% above the control's for 10 Myr (paired, same seed).
 2. Plate events by 50 Myr window (stats.json event_list, saved since the thirty-ninth pass): rift, detach, merge,
    subduction initiation; detached area.
 3. Where the trench is (snapshots every 10 Myr, then 1 Myr): subduction nodes within 150 km of small continental
    bodies (connected continental crust < 2 Mkm^2: microcontinents and arcs turned continental), of thin continental
    crust (< 25 km), of oceanic arc crust (> 10 km), of the large continents, or far from all.
 4. Small continental bodies: count, area, thickness p10 / p50 / p90 (Earth's island arcs 26 +- 6 km, Tetreault &
    Buiter 2014; the plate layer turns arcs continental at 20 km), share thinner than 20 km.
 5. Spikes: oceanic nodes > 12 km among >= 5 neighbours < 10 km; continental nodes with <= 1 continental neighbour.
 6. Detach events in the last 40 Myr (1 Myr snapshots): the piece, its parent and the plate that cut it, with the
    cutter's crust where it cut (continental share, thickness, in a small continental body or not).
 7. Arc flux over time (series): arc growth, docked on continents, docked and spread on oceanic plates, turned
    continental (area ledger), km^3/yr per 50 Myr window.
Writes <sweep>/reviews/arc_dock_probe.json and arc_dock_probe.png (trench and plates against time, three arms, per seed).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import graph_landscape as gl
from planet_layer.plates import SUBDUCTION
from planet_layer.sphere_graph import CubeSphere

try:
    from numba import njit
except ImportError:                                   # pure Python fallback (slow)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = [(-250, -200), (-200, -150), (-150, -100), (-100, -50), (-50, 0.1)]
SMALL_MKM2 = 2.
NEAR_M = 150e3
KINDS = ('rift', 'detach', 'merge', 'subduction initiation')


def snap(sd, t):
    return np.load(Path(sd)/'snapshots'/f't{float(t):+07.1f}.npz')


def snap_times(sd):
    return sorted(float(f.stem[1:]) for f in (Path(sd)/'snapshots').glob('t*.npz'))


@njit(cache=True)
def _bfs_labels(mask, nbr):
    n = mask.shape[0]
    lab = np.full(n, -1, np.int64)
    stack = np.empty(n, np.int64)
    for s in range(n):
        if not mask[s] or lab[s] >= 0:
            continue
        lab[s] = s; top = 0; stack[0] = s; top = 1
        while top > 0:
            top -= 1
            i = stack[top]
            for k in range(nbr.shape[1]):
                j = nbr[i, k]
                if j >= 0 and mask[j] and lab[j] < 0:
                    lab[j] = s; stack[top] = j; top += 1
    return lab


def components(mask, nbr):
    """Connected components of mask (labels, -1 outside)."""
    return _bfs_labels(np.ascontiguousarray(mask, np.bool_), np.ascontiguousarray(nbr, np.int64))


def snapshot_reading(g, z):
    c = z['continental'].astype(bool)
    T = z['crust_thickness_m'].astype(np.float64)
    bt = z['boundary_type']
    a = g.area
    lab = components(c, g.nbr)
    ids, inv = np.unique(lab[c], return_inverse=True)
    area = np.bincount(inv, weights=a[c])
    small = np.zeros(g.size, bool)
    small[np.flatnonzero(c)] = area[inv] < SMALL_MKM2*1e12
    big = c & ~small
    thin = c & (T < 25e3)
    arc = (~c) & (T > 10e3)
    sub = bt == SUBDUCTION
    d = {k: gl.distance_to(m, g.nbr, g.dist, 2*NEAR_M) if m.any() else np.full(g.size, np.inf)
         for k, m in (('small', small), ('thin', thin), ('arc', arc), ('big', big))}
    near_small = sub & (d['small'] < NEAR_M)
    near_thin = sub & (d['thin'] < NEAR_M) & ~near_small
    near_arc = sub & (d['arc'] < NEAR_M) & ~near_small & ~near_thin
    near_big = sub & (d['big'] < NEAR_M) & ~near_small & ~near_thin & ~near_arc
    far = sub & ~(near_small | near_thin | near_arc | near_big)
    h = np.sqrt(a.mean())/1e3                                    # km per node; both sides counted: ~2-3x the line
    ok = g.nbr >= 0
    nb = np.where(ok, g.nbr, 0)
    spikes = (~c) & (T > 12e3) & ((ok & (T[nb] < 10e3)).sum(1) >= 5)
    lone = c & ((ok & c[nb]).sum(1) <= 1)
    Ts = T[small]
    q = lambda x, p: float(np.percentile(x, p)/1e3) if x.size else float('nan')
    return {'trench_node_km': float(sub.sum()*h),
            'trench_km_near_small_continents': float(near_small.sum()*h), 'trench_km_near_thin_continent': float(near_thin.sum()*h),
            'trench_km_near_ocean_arcs': float(near_arc.sum()*h), 'trench_km_near_large_continents': float(near_big.sum()*h),
            'trench_km_far': float(far.sum()*h),
            'small_bodies': int((area < SMALL_MKM2*1e12).sum()), 'small_area_frac': float(a[small].sum()/a.sum()),
            'small_thick_p10_p50_p90_km': [q(Ts, 10), q(Ts, 50), q(Ts, 90)],
            'small_share_under_20km': float(a[small & (T < 20e3)].sum()/max(a[small].sum(), 1.)),
            'continental_under_20km_frac': float(a[c & (T < 20e3)].sum()/a.sum()),
            'ocean_arc_area_frac': float(a[arc].sum()/a.sum()),
            'spikes': int(spikes.sum()), 'lone_continental_nodes': int(lone.sum()), 'plates': int(np.unique(z['plate_id']).size)}


def detach_cutters(g, sd, events):
    """Detach events of the last 40 Myr: the plate that took the nodes that separated the piece from its parent."""
    out = []
    times = set(snap_times(sd))
    for e in events:
        if e['kind'] != 'detach' or e['t'] < -39.5:
            continue
        t1 = round(e['t']); t0 = t1-1
        if float(t0) not in times or float(t1) not in times:
            continue
        z0, z1 = snap(sd, t0), snap(sd, t1)
        p0, p1 = z0['plate_id'], z1['plate_id']
        piece = p1 == e['new_plate']
        par = e['plate']
        if not piece.any():
            continue
        ring = piece.copy()
        ok = g.nbr >= 0; nb = np.where(ok, g.nbr, 0)
        for _ in range(4):
            ring = ring | np.any(ok & ring[nb], 1)
        took = ring & ~piece & (p0 == par) & (p1 != par)          # the parent's nodes another plate took
        if not took.any():
            out.append({'t': e['t'], 'piece_mkm2': e.get('area_mkm2'), 'cutter': None})
            continue
        cutter = int(np.bincount(p1[took]).argmax())
        m = took & (p1 == cutter)
        c1 = z1['continental'].astype(bool); T1 = z1['crust_thickness_m'].astype(np.float64)
        lab = components(c1, g.nbr)
        ids, inv = np.unique(lab[c1], return_inverse=True)
        area = np.bincount(inv, weights=g.area[c1])
        small = np.zeros(g.size, bool); small[np.flatnonzero(c1)] = area[inv] < SMALL_MKM2*1e12
        out.append({'t': e['t'], 'piece_mkm2': e.get('area_mkm2'), 'parent': par, 'cutter': cutter,
                    'cut_nodes': int(m.sum()), 'cutter_continental_share': float(c1[m].mean()),
                    'cutter_thick_p50_km': float(np.median(T1[m])/1e3), 'cutter_in_small_continent': float(small[m].mean()),
                    'piece_continental_share': float(c1[piece].mean())})
    return out


def series_rates(ser, key, lo, hi):
    """km^3/yr of an erosion total over a window (series values are km^3 totals)."""
    v = [(r['time_myr'], r.get('erosion_totals_km3', {}).get(key, 0.)) for r in ser]
    a = [x for x in v if x[0] >= lo-1e-6 and x[0] <= hi+1e-6]
    if len(a) < 2:
        return float('nan')
    return (a[-1][1]-a[0][1])/((a[-1][0]-a[0][0])*1e6)


def area_rate(ser, key, lo, hi, A_km2):
    v = [(r['time_myr'], r.get('area_ledger_frac', {}).get(key, 0.)) for r in ser]
    a = [x for x in v if lo-1e-6 <= x[0] <= hi+1e-6]
    return (a[-1][1]-a[0][1])*A_km2/((a[-1][0]-a[0][0])*1e6) if len(a) > 1 else float('nan')


def seed_reading(g, sd, ctrl_sd=None):
    st = json.loads((Path(sd)/'stats.json').read_text())
    ser = st['series']
    t = np.array([r['time_myr'] for r in ser])
    tr = np.array([r['subduction_km'] for r in ser])
    out = {'series': {'t': t.tolist(), 'trench_km': tr.tolist(), 'ridge_km': [r['ridge_km'] for r in ser],
                      'plates': [r['plates'] for r in ser],
                      'speed_mean_cm_yr': [float(np.mean(r['speeds_cm_yr'])) for r in ser]}}
    if ctrl_sd is not None:
        cs = json.loads((Path(ctrl_sd)/'stats.json').read_text())['series']
        ct = {r['time_myr']: r['subduction_km'] for r in cs}
        above = np.array([tr[i] >= 1.15*ct.get(t[i], np.inf) for i in range(len(t))])
        first = next((float(t[i]) for i in range(len(t)-10) if above[i:i+10].all()), None)
        out['trench_15pct_above_control_from_myr'] = first
    ev = st.get('event_list')
    out['have_event_list'] = ev is not None
    ev = ev or []
    out['events_by_window'] = {f'{lo:.0f}..{min(hi, 0):.0f}': {k: sum(1 for e in ev if e['kind'] == k and lo <= e['t'] < hi) for k in KINDS}
                               for lo, hi in WINDOWS}
    out['detached_area_mkm2'] = float(sum(e.get('area_mkm2', 0.) for e in ev if e['kind'] == 'detach'))
    A_km2 = 4*np.pi*(g.radius/1e3)**2
    out['arc_flux_by_window_km3_yr'] = {
        f'{lo:.0f}..{min(hi, 0):.0f}': {'growth': series_rates(ser, 'subduction_on_ocean_arcs_m3', lo, min(hi, 0)),
                                        'docked_on_continents': series_rates(ser, 'arc_accreted_on_continent_m3', lo, min(hi, 0)),
                                        'docked_spread_on_ocean': series_rates(ser, 'arc_docked_spread_m3', lo, min(hi, 0)),
                                        'turned_continental_mkm2_per_myr': area_rate(ser, 'arc', lo, min(hi, 0), A_km2)*1e6/1e6}
        for lo, hi in WINDOWS}
    times = snap_times(sd)
    pick = [x for x in times if x <= -40 and abs(x % 10) < 1e-6]+[x for x in (-30., -20., -10., 0.) if x in times]
    line = {r['time_myr']: r['subduction_km'] for r in ser}
    out['line_trench_km'] = {f'{x:.0f}': line.get(x, float('nan')) for x in pick}
    out['snapshots'] = {f'{x:.0f}': snapshot_reading(g, snap(sd, x)) for x in pick}
    out['detach_cutters_last40'] = detach_cutters(g, sd, ev)
    return out


def seeds_of(d):
    d = Path(d)
    return sorted(x for x in d.glob('seed*') if (x/'snapshots').is_dir())


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float('nan')


def report(name, res):
    S = list(res.values())
    print(f"\n=== {name} ({len(S)} seeds) ===")
    first = [r.get('trench_15pct_above_control_from_myr') for r in S]
    if any(k in S[0] for k in ('trench_15pct_above_control_from_myr',)):
        print("  trench >= 15% above the control's for 10 Myr, from (Myr, by seed): "
              + ' / '.join('-' if f is None else f'{f:.0f}' for f in first))
    print("  events by window (rift / detach / merge / subduction initiation; six-seed sums):")
    for w in S[0]['events_by_window']:
        c = {k: sum(r['events_by_window'][w][k] for r in S) for k in KINDS}
        print(f"    {w:>10}: " + ' / '.join(str(c[k]) for k in KINDS))
    if not all(r['have_event_list'] for r in S):
        print("    (runs before the thirty-ninth pass have no event_list: counts are 0)")
    print("  arc flux by window, km^3/yr (growth / docked on continents / docked spread on ocean), six-seed means:")
    for w in S[0]['arc_flux_by_window_km3_yr']:
        f = lambda k: mean([r['arc_flux_by_window_km3_yr'][w][k] for r in S])
        print(f"    {w:>10}: {f('growth'):.2f} / {f('docked_on_continents'):.2f} / {f('docked_spread_on_ocean'):.2f}")
    print("  snapshots (six-seed means): trench node-km (subduction-typed nodes on both sides x cell width: ~2-3x the "
          "line length; read as shares) near small continents / thin continent / ocean arcs / large "
          "continents / far | small bodies: count, area, T p10/p50/p90 km, share < 20 km | spikes, lone continental nodes, plates")
    for x in S[0]['snapshots']:
        if not all(x in r['snapshots'] for r in S):
            continue
        m = lambda k: mean([r['snapshots'][x][k] for r in S])
        tq = [mean([r['snapshots'][x]['small_thick_p10_p50_p90_km'][i] for r in S]) for i in range(3)]
        print(f"    {x:>5}: {m('trench_node_km')/1e3:5.1f}k (line {mean([r['line_trench_km'].get(x, np.nan) for r in S])/1e3:5.1f}k) = {m('trench_km_near_small_continents')/1e3:4.1f} / "
              f"{m('trench_km_near_thin_continent')/1e3:4.1f} / {m('trench_km_near_ocean_arcs')/1e3:4.1f} / "
              f"{m('trench_km_near_large_continents')/1e3:4.1f} / {m('trench_km_far')/1e3:4.1f} | {m('small_bodies'):5.1f}, "
              f"{m('small_area_frac'):.3f}, {tq[0]:.1f}/{tq[1]:.1f}/{tq[2]:.1f}, {m('small_share_under_20km'):.2f} | "
              f"{m('spikes'):5.0f}, {m('lone_continental_nodes'):5.0f}, {m('plates'):4.1f}")
    cut = [c for r in S for c in r['detach_cutters_last40'] if c.get('cutter') is not None]
    if cut:
        f = lambda k: mean([c[k] for c in cut])
        print(f"  detach events in the last 40 Myr with a cutter: {len(cut)} (six seeds); cutter where it cut: continental "
              f"share {f('cutter_continental_share'):.2f}, thickness p50 {f('cutter_thick_p50_km'):.1f} km, in a small "
              f"continental body {f('cutter_in_small_continent'):.2f}; piece {f('piece_mkm2'):.1f} Mkm^2, continental share "
              f"{f('piece_continental_share'):.2f}")
    else:
        print("  detach events in the last 40 Myr with a cutter: none")


def figure(path, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seeds = sorted({s for arm in data.values() for s in arm})
    fig, ax = plt.subplots(2, len(seeds), figsize=(3.2*len(seeds), 6), sharex=True, squeeze=False)
    for j, s in enumerate(seeds):
        for name, arm in data.items():
            if s not in arm:
                continue
            se = arm[s]['series']
            ax[0, j].plot(se['t'], np.array(se['trench_km'])/1e3, label=name, lw=1)
            ax[1, j].plot(se['t'], se['plates'], lw=1)
        ax[0, j].set_title(s)
        ax[1, j].set_xlabel('Myr')
    ax[0, 0].set_ylabel('trench, node length (k km)')
    ax[1, 0].set_ylabel('plates')
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep', nargs='?')
    ap.add_argument('--latest', action='store_true')
    ap.add_argument('--control')
    a = ap.parse_args(argv)
    sw = Path(a.sweep) if a.sweep else sorted((ROOT/'outputs'/'planet-calibration').glob('sweep-*'))[-1]
    print('sweep', sw)
    arms = sorted(d for d in sw.iterdir() if d.is_dir() and d.name != 'reviews' and seeds_of(d))
    ctrl = next((d for d in arms if (a.control and d.name == a.control) or
                 (not a.control and 'arc_H_ocean_m-300' in d.name and 'arc_accretion-False' in d.name)), None)
    print('control', ctrl.name if ctrl else '(none)')
    g = CubeSphere(217)
    data = {}
    for arm in ([ctrl] if ctrl else [])+[d for d in arms if d != ctrl]:
        res = {}
        for sd in seeds_of(arm):
            csd = ctrl/sd.name if ctrl and arm != ctrl and (ctrl/sd.name).is_dir() else None
            res[sd.name] = seed_reading(g, sd, csd)
            print(f"  read {arm.name}/{sd.name}", flush=True)
        data[arm.name] = res
        report(arm.name, res)
    (sw/'reviews').mkdir(exist_ok=True)
    (sw/'reviews'/'arc_dock_probe.json').write_text(json.dumps(data, indent=1, default=float))
    figure(sw/'reviews'/'arc_dock_probe.png', data)
    print('written', sw/'reviews'/'arc_dock_probe.json', 'and .png')


if __name__ == '__main__':
    main()
