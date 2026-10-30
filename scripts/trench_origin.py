"""Why ocean-ocean convergence is so common on the planet (item B-b, step 1; plan doc, thirty-sixth pass).

  python -m scripts.trench_origin outputs\\planet-calibration\\sweep-20261025-170555\\boundary_normal-ring_extension-False_ext_onset_thick_m-45000

Reads only saved snapshots (t<time>.npz with t<time>-plates.json); changes nothing in the model. For a combination
folder (all its seeds) or seed folders, at -250, -240, -200, -150, -100, -50, -20 and 0 Myr:

1. Boundary geometry: total contour length (scripts.boundary_lengths) against the sum of the half perimeters of
   spherical caps with the plates' areas (the shortest boundary those plate areas allow); the plates' detached
   pieces (connected parts other than each plate's largest) and the boundary length around them.
2. Ocean-ocean boundary (oceanic crust on both sides, the continental flag) by type; intra-oceanic subduction as a
   share of it; its closing-rate distribution (length weighted) and the share slower than 20 mm/yr.
3. Area fluxes: ridge production and consumption (km^2/yr: length x opening or closing rate) at intra-oceanic and
   continental-side trenches and at collisions.
4. Plate speeds against the plate layer's speed law (plates.PlateSim._driving's target, computed from the same
   node counts the law reads) and against Earth's plates at the same slab fraction of the perimeter.
5. Intra-oceanic trench by plate pair at 0 Myr: length, rate, both plates' continental share and speed, the
   plates' origin (start plate or created later), and since when the pair has had intra-oceanic trench.

Earth (computed here, cached in outputs/planet-reference/earth-trenches.json): Bird 2003 PB2002_steps (5,824 steps:
class, length, divergence / convergence rate), each step's crust from CRUST1.0 types 100 and 200 km to both sides
(intra-oceanic: types A0, A1, B-, V1, island arc J- and forearc K- on both sides at both distances); Bird's
subduction steps name the subducting plate by the slash (A/B: B subducts; A\\B: A subducts); plate NNR speeds from
outputs/planet-reference/earth-plate-speeds-nnr.json (thirteenth pass). Bird's OCB class (oceanic convergent
boundary without a slab, 17.4k km) is shown beside SUB: on Earth that convergence makes no trench.
"""
import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from planet_layer import plates as P
from planet_layer.plates import components
from planet_layer.sphere_graph import CubeSphere
from scripts import boundary_lengths as bl
from scripts import plate_speeds as ps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
CACHE = ROOT/'outputs'/'planet-reference'/'earth-trenches.json'
TIMES = (-250., -240., -200., -150., -100., -50., -20., 0.)
OCEANIC_TYPES = ('A0', 'A1', 'B-', 'V1', 'J-', 'K-')
SLOW_MM_YR = 20.
FRACTION_BINS = ((0., .05), (.05, .15), (.15, .25), (.25, .4), (.4, 1.01))
R_KM = 6371.


def read_dbf(path):
    """dBase III table (Bird's .dbf files) as {field: array}; numbers as float, text stripped."""
    raw = Path(path).read_bytes()
    n, hl, rl = struct.unpack('<IHH', raw[4:12])
    fl, o = [], 32
    while raw[o] != 0x0d:
        d = raw[o:o+32]
        fl.append((d[:11].split(b'\0')[0].decode(), chr(d[11]), d[16])); o += 32
    out = {k: [] for k, _, _ in fl}
    for i in range(n):
        r, q = raw[hl+i*rl+1:hl+(i+1)*rl], 0
        for k, t, w in fl:
            s = r[q:q+w].decode('latin1').strip(); q += w
            out[k].append((float(s) if s else np.nan) if t == 'N' else s)
    return {k: np.array(v) for k, v in out.items()}


def wpct(v, w, qs):
    """Length-weighted percentiles."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    if not v.size or w.sum() <= 0:
        return [float('nan')]*len(qs)
    o = np.argsort(v); c = np.cumsum(w[o])
    return [float(v[o][min(np.searchsorted(c, c[-1]*q), v.size-1)]) for q in qs]


def cap_perimeter_km(area_km2):
    c = 1-np.asarray(area_km2, float)/(2*np.pi*R_KM**2)
    return 2*np.pi*R_KM*np.sin(np.arccos(np.clip(c, -1, 1)))


def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], -1)


def speed_target(L_slab, L_ridge, L_coll, cf, p=P.PARAMS):
    """The plate layer's target speed (cm/yr), plates.PlateSim._driving, from the lengths (km) it reads."""
    t = (p['v_min_cm_yr']+(p['v_slab_cm_yr']-p['v_min_cm_yr'])*np.minimum(1., L_slab/p['slab_ref_km'])
         + p['v_ridge_cm_yr']*np.minimum(1., L_ridge/p['ridge_ref_km']))
    return t*(1-p['cont_drag']*cf)/(1+L_coll/p['collision_ref_km'])


def earth_reference(refresh=False):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    s = read_dbf(DATA/'tectonicplates-master'/'PB2002_steps.dbf')
    types = np.array([l.split() for l in (DATA/'crust1'/'CNtype1-1.txt').read_text().splitlines() if l.strip()])
    a = _xyz(s['STARTLONG'], s['STARTLAT']); b = _xyz(s['FINALLONG'], s['FINALLAT'])
    m = a+b; m /= np.linalg.norm(m, axis=1, keepdims=True)
    nrm = np.cross(m, b-a); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    def oceanic(dist_km):
        ok = np.ones(len(m), bool)
        for sg in (1, -1):
            for d in (dist_km/2, dist_km):
                q = m+sg*d/R_KM*nrm; q /= np.linalg.norm(q, axis=1, keepdims=True)
                lat = np.degrees(np.arcsin(np.clip(q[:, 2], -1, 1))); lon = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
                r = np.clip((89.5-lat+.5).astype(int), 0, 179); c = np.clip((lon+180).astype(int), 0, 359)
                ok &= np.isin(types[r, c], OCEANIC_TYPES)
        return ok
    L, cl, dv = s['STEPLENGTH'], s['STEPCLASS'], s['VELOCITYDI']
    out = {'source': 'Bird 2003 PB2002_steps.dbf; CRUST1.0 CNtype1-1.txt (oceanic, island-arc and forearc types on both '
                     'sides); plate speeds from earth-plate-speeds-nnr.json', 'classes': {}}
    for c in np.unique(cl):
        k = cl == c
        sgn = 1. if c in ('OSR', 'CRB') else -1.
        out['classes'][c] = {'km': float(L[k].sum()), 'rate_p10_p50_p90_mm_yr': wpct(sgn*dv[k], L[k], (.1, .5, .9)),
                             'flux_km2_yr': float((L[k]*sgn*dv[k]).sum()*1e-6)}
    sub = cl == 'SUB'
    for d in (100, 200):
        io = sub & oceanic(d)
        out[f'sub_intra_oceanic_km_{d}'] = float(L[io].sum())
    io = sub & oceanic(200)
    for name, k in (('sub_intra_oceanic', io), ('sub_continental', sub & ~io), ('ocb', cl == 'OCB')):
        v = -dv[k]
        out[name] = {'km': float(L[k].sum()), 'rate_p10_p50_p90_mm_yr': wpct(v, L[k], (.1, .5, .9)),
                     'slow_share': float(L[k & (-dv < SLOW_MM_YR)].sum()/L[k].sum()),
                     'flux_km2_yr': float((L[k]*v).sum()*1e-6)}
    oo = io | np.isin(cl, ('OSR', 'OTF', 'OCB'))
    out['ocean_ocean_km'] = float(L[oo].sum())
    out['total_km'] = float(L.sum())
    # plates: slab (downgoing SUB), ridge (OSR, CRB), collision (CCB), perimeter
    slab, ridge, coll, per = Counter(), Counter(), Counter(), Counter()
    for pb, l, c in zip(s['PLATEBOUND'], L, cl):
        if '/' in pb:
            x, y = pb.split('/'); down = y
        elif '\\' in pb:
            x, y = pb.split('\\'); down = x
        else:
            x, y = pb.split('-'); down = None
        per[x] += l; per[y] += l
        if c == 'SUB':
            slab[down] += l
        elif c in ('OSR', 'CRB'):
            ridge[x] += l; ridge[y] += l
        elif c == 'CCB':
            coll[x] += l; coll[y] += l
    e = json.loads((ROOT/'outputs'/'planet-reference'/'earth-plate-speeds-nnr.json').read_text())['plates']
    out['plates'] = [{'id': r['id'], 'area_mkm2': r['area_mkm2'], 'speed_cm_yr': r['rms_speed_cm_yr'],
                      'continental_share': r['continental_share'], 'slab_km': slab[r['id']], 'ridge_km': ridge[r['id']],
                      'collision_km': coll[r['id']], 'perimeter_km': per[r['id']]} for r in e]
    out['cap_half_perimeters_km'] = float(cap_perimeter_km(np.array([r['area_mkm2'] for r in e])*1e6).sum()/2)
    CACHE.write_text(json.dumps(out, indent=1))
    return out


def snap_path(seed_dir, t):
    return Path(seed_dir)/'snapshots'/('t+0000.0' if t == 0 else f't{t:+07.1f}')


def node_lengths(g, z, plates):
    """Per plate, the lengths the speed law reads (km): own trench nodes, ridge nodes, collision nodes (halved node
    counts x mean spacing, as plates.PlateSim._driving), and the plate's continental share."""
    pid = z['plate_id'].astype(np.int64)
    ids = np.array([r['id'] for r in plates])
    slot = np.zeros(max(ids.max(), pid.max())+1, np.int64); slot[ids] = np.arange(len(ids))
    omega = np.stack([np.asarray(r['axis'], float)*r['speed_cm_yr']*1e-2/g.radius for r in plates])
    cont = z['continental'].astype(bool)
    share = np.array([g.area[(pid == k) & cont].sum() for k in ids])/g.area.sum()
    age = np.nan_to_num(z['ocean_age_myr'].astype(float))
    score = np.where(cont, 1e6+1e5*share[slot[pid]], -age)
    bt, rate, trench = P.classify_boundaries_ring(pid, slot, omega, g.xyz, g.nbr, score, cont, g.radius,
                                                  P.PARAMS['boundary_min_mm_yr']*1e-3)
    h = np.sqrt(g.area.mean())/1e3
    out = {}
    for k in ids:
        own = pid == k
        out[int(k)] = {'slab': (own & trench).sum()*h/2, 'ridge': (own & (bt == P.RIDGE)).sum()*h/2,
                       'coll': (own & (bt == P.COLLISION)).sum()*h/2,
                       'cf': float(g.area[own & cont].sum()/max(g.area[own].sum(), 1.))}
    return out


def snapshot_metrics(g, f):
    z = np.load(str(f)+'.npz'); plates = json.loads(Path(str(f)+'-plates.json').read_text())
    pid = z['plate_id']; cont = z['continental'].astype(bool); age = np.nan_to_num(z['ocean_age_myr'].astype(float))
    main = np.zeros(g.size, bool)
    areas = []
    for k in np.unique(pid):
        m = pid == k
        lab = components(m, g.nbr)
        sz = np.bincount(lab[lab >= 0], weights=g.area[lab >= 0])
        main |= m & (lab == np.argmax(sz))
        areas.append(g.area[m].sum()/1e6)
    s = bl.typed_segments(g, pid, plates, cont)
    L, typ, cv, na, nb, ka, kb = s['L'], s['typ'], s['conv'], s['na'], s['nb'], s['ka'], s['kb']
    oo = ~(cont[na] | cont[nb])
    frag = ~(main[na] & main[nb])
    io = (typ == 2) & oo
    cs = (typ == 2) & ~oo
    ri = typ == 1
    r = {'t': float(z['time_myr']), 'plates': len(plates), 'total_km': float(L.sum()),
         'cap_half_perimeters_km': float(cap_perimeter_km(np.array(areas)).sum()/2),
         'detached_mkm2': float(g.area[~main].sum()/1e12), 'detached_boundary_km': float(L[frag].sum()),
         'ocean_ocean_km': float(L[oo & (typ > 0)].sum()), 'ridge_km': float(L[ri].sum()),
         'io_km': float(L[io].sum()), 'io_detached_km': float(L[io & frag].sum()), 'cont_sub_km': float(L[cs].sum()),
         'io_rate_p10_p50_p90': wpct(cv[io], L[io], (.1, .5, .9)),
         'io_slow_share': float(L[io & (cv < SLOW_MM_YR)].sum()/max(L[io].sum(), 1.)),
         'ridge_rate_p50': wpct(-cv[ri], L[ri], (.5,))[0],
         'production_km2_yr': float((L[ri]*-cv[ri]).sum()*1e-6), 'consumption_io_km2_yr': float((L[io]*cv[io]).sum()*1e-6),
         'consumption_cont_km2_yr': float((L[cs]*cv[cs]).sum()*1e-6),
         'consumption_coll_km2_yr': float((L[typ == 3]*cv[typ == 3]).sum()*1e-6)}
    r['shape_ratio'] = r['total_km']/r['cap_half_perimeters_km']
    r['io_share_of_ocean_ocean'] = r['io_km']/max(r['ocean_ocean_km'], 1.)
    # speeds: NNR, the law's target, slab fraction of the perimeter (contour: downgoing side = lower score)
    sa = np.where(cont[na], 1e6, -age[na]); sb = np.where(cont[nb], 1e6, -age[nb])
    down = np.where(sa < sb, ka, kb)
    om = {int(q['id']): np.asarray(q['axis'], float)*q['speed_cm_yr']*1e-2/g.radius for q in plates}
    _, rows = ps.nnr_speeds(g, pid, om, cont.astype(float))
    nl = node_lengths(g, z, plates)
    sp = {int(q['id']): q['speed_cm_yr'] for q in plates}
    pr = []
    for k, A, v, cf in rows:
        per = float(L[(ka == k) | (kb == k)].sum())
        slab = float(L[(typ == 2) & (down == k)].sum())
        q = nl[k]
        pr.append({'id': k, 'area_mkm2': A, 'nnr_cm_yr': v, 'speed_cm_yr': sp[k], 'cf': cf, 'slab_km': slab,
                   'perimeter_km': per, 'slab_fraction': slab/max(per, 1.),
                   'target_cm_yr': float(speed_target(q['slab'], q['ridge'], q['coll'], q['cf'])),
                   'node_slab_km': float(q['slab'])})
    r['plates_rows'] = pr
    # pairs of intra-oceanic trench
    pairs = Counter()
    for a_, b_, l in zip(ka[io], kb[io], L[io]):
        pairs[tuple(sorted((int(a_), int(b_))))] += float(l)
    r['io_pairs'] = {f'{a_}-{b_}': l for (a_, b_), l in pairs.items()}
    return r


def seed_dirs(dirs):
    out = []
    for d in dirs:
        d = Path(d)
        out += sorted(d.glob('seed*')) if not (d/'snapshots').exists() else [d]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', help='combination folders (all seeds) or seed folders')
    ap.add_argument('--refresh-earth', action='store_true')
    ap.add_argument('--json', help='write all numbers here')
    a = ap.parse_args(argv)
    e = earth_reference(a.refresh_earth)
    C = e['classes']
    print('Earth (Bird 2003 + CRUST1.0):')
    print(f"  boundary total {e['total_km']/1e3:.1f}k km over 52 plates; cap half-perimeters {e['cap_half_perimeters_km']/1e3:.1f}k;"
          f" shape ratio {e['total_km']/e['cap_half_perimeters_km']:.2f}")
    print(f"  subduction {C['SUB']['km']/1e3:.1f}k km: intra-oceanic {e['sub_intra_oceanic']['km']/1e3:.1f}k (types 200 km to"
          f" both sides; 100 km: {e['sub_intra_oceanic_km_100']/1e3:.1f}k), continental side {e['sub_continental']['km']/1e3:.1f}k;"
          f" Leat & Larter 2003: ~40% of subduction margins")
    for name in ('sub_intra_oceanic', 'sub_continental', 'ocb'):
        q = e[name]
        print(f"  {name}: closing p10/p50/p90 {'/'.join(f'{x:.0f}' for x in q['rate_p10_p50_p90_mm_yr'])} mm/yr, slower than"
              f" {SLOW_MM_YR:.0f}: {q['slow_share']:.2f}, consumption {q['flux_km2_yr']:.2f} km^2/yr")
    prod = C['OSR']['flux_km2_yr']+C['CRB']['flux_km2_yr']
    print(f"  ridges OSR {C['OSR']['km']/1e3:.1f}k + CRB {C['CRB']['km']/1e3:.1f}k km, opening p50 OSR"
          f" {C['OSR']['rate_p10_p50_p90_mm_yr'][1]:.0f} mm/yr; production {prod:.2f} km^2/yr (Rowley 2002: 3.4, steady since"
          f" 180 Ma); ocean-ocean boundary {e['ocean_ocean_km']/1e3:.1f}k km, intra-oceanic subduction"
          f" {e['sub_intra_oceanic']['km']/e['ocean_ocean_km']:.2f} of it, with OCB {(e['sub_intra_oceanic']['km']+e['ocb']['km'])/e['ocean_ocean_km']:.2f}")
    ep = [q for q in e['plates'] if q['area_mkm2'] > 3]
    print('  plates > 3 Mkm^2 (id, NNR speed cm/yr, continental share, slab fraction of the perimeter):')
    print('   ' + '; '.join('%s %.1f cf %.2f slab %.2f' % (q['id'], q['speed_cm_yr'], q['continental_share'],
                                                          q['slab_km']/q['perimeter_km'])
                            for q in sorted(ep, key=lambda q: -q['area_mkm2'])))
    allr = {}
    g = CubeSphere(217)
    for sd in seed_dirs(a.dirs):
        rows = []
        for t in TIMES:
            f = snap_path(sd, t)
            if Path(str(f)+'.npz').exists():
                rows.append(snapshot_metrics(g, f))
        allr[str(sd)] = rows
    # node factor: law's slab length / contour slab length, on the planet
    fac = [q['node_slab_km']/q['slab_km'] for rows in allr.values() for r in rows for q in r['plates_rows'] if q['slab_km'] > 2000]
    fnode = float(np.median(fac)) if fac else float('nan')
    print(f"\n  node factor (law's trench length / contour trench length, plates with > 2,000 km): median {fnode:.2f}"
          f" (p10-p90 {np.percentile(fac, 10):.2f}-{np.percentile(fac, 90):.2f})")
    print('  the law on Earth\'s plates (Bird lengths x node factor; continental share as Earth\'s):')
    tl = []
    for q in sorted(ep, key=lambda q: -q['area_mkm2']):
        tq = float(speed_target(q['slab_km']*fnode, q['ridge_km']*fnode, q['collision_km']*fnode, q['continental_share']))
        tl.append((q['id'], q['speed_cm_yr'], tq))
    print('   ' + '; '.join(f'{i} {v:.1f} -> {tq:.1f}' for i, v, tq in tl))
    print('   Earth speed / law, area-weighted over these plates: '
          f"{sum(q['area_mkm2']*q['speed_cm_yr'] for q in ep)/sum(q['area_mkm2']*t for q, (_, _, t) in zip(sorted(ep, key=lambda q: -q['area_mkm2']), tl)):.2f}")
    for sd, rows in allr.items():
        print(f'\n{sd}')
        print('   t      plates shape  detached(Mkm2/bnd k) oo(k)  io(k) io-det  io/oo  io p10/50/90  slow  cont-sub(k)'
              '  ridge(k) p50  prod  cons io/cont/coll (km^2/yr)')
        for r in rows:
            print(f"  {r['t']:6.0f} {r['plates']:4d}  {r['shape_ratio']:5.2f}  {r['detached_mkm2']:5.1f}/{r['detached_boundary_km']/1e3:5.1f}"
                  f"  {r['ocean_ocean_km']/1e3:6.1f} {r['io_km']/1e3:5.1f}  {r['io_detached_km']/1e3:5.1f}  {r['io_share_of_ocean_ocean']:.2f}"
                  f"  {'/'.join(f'{x:.0f}' for x in r['io_rate_p10_p50_p90'])}  {r['io_slow_share']:.2f}  {r['cont_sub_km']/1e3:5.1f}"
                  f"  {r['ridge_km']/1e3:6.1f} {r['ridge_rate_p50']:3.0f}  {r['production_km2_yr']:.2f}  {r['consumption_io_km2_yr']:.2f}"
                  f"/{r['consumption_cont_km2_yr']:.2f}/{r['consumption_coll_km2_yr']:.2f}")
    # speeds against the law and against Earth at the same slab fraction
    print('\nplates > 10 Mkm^2, all seeds and times from -200 Myr: NNR speed by slab fraction of the perimeter (median, n)')
    pl = [q for rows in allr.values() for r in rows if r['t'] >= -200 for q in r['plates_rows'] if q['area_mkm2'] > 10]
    for lo, hi in FRACTION_BINS:
        def med(qs, key): return f"{np.median([q[key] for q in qs]):.1f}" if qs else '  - '
        po = [q for q in pl if lo <= q['slab_fraction'] < hi and q['cf'] <= .45]
        pc = [q for q in pl if lo <= q['slab_fraction'] < hi and q['cf'] > .45]
        eo = [q for q in e['plates'] if q['area_mkm2'] > 3 and lo <= q['slab_km']/q['perimeter_km'] < hi and q['continental_share'] <= .45]
        ec = [q for q in e['plates'] if q['area_mkm2'] > 3 and lo <= q['slab_km']/q['perimeter_km'] < hi and q['continental_share'] > .45]
        def ids(qs): return ', '.join('%s %.1f' % (q['id'], q['speed_cm_yr']) for q in qs) or '-'
        print(f"  {lo:.2f}-{hi:.2f}: oceanic planet {med(po, 'nnr_cm_yr')} (n {len(po)}), Earth {ids(eo)};"
              f" continental planet {med(pc, 'nnr_cm_yr')} (n {len(pc)}), Earth {ids(ec)}")
    sp = np.array([q['speed_cm_yr'] for q in pl]); tg = np.array([q['target_cm_yr'] for q in pl])
    print(f"  planet speed / law target: median {np.median(sp/tg):.2f} (p10-p90 {np.percentile(sp/tg, 10):.2f}-{np.percentile(sp/tg, 90):.2f})")
    print(f"  planet perimeter / cap perimeter, plates > 10 Mkm^2: median "
          f"{np.median([q['perimeter_km']/cap_perimeter_km(q['area_mkm2']*1e6) for q in pl]):.2f}; Earth "
          f"{np.median([q['perimeter_km']/cap_perimeter_km(q['area_mkm2']*1e6) for q in e['plates'] if q['area_mkm2'] > 10]):.2f}")
    # pairs at 0 Myr, with persistence
    print('\nintra-oceanic trench by plate pair at 0 Myr (k km; first listed time with >= 500 km of it; start plates are ids < 12):')
    for sd, rows in allr.items():
        last = rows[-1]
        pr = {q['id']: q for q in last['plates_rows']}
        top = sorted(last['io_pairs'].items(), key=lambda x: -x[1])[:6]
        txt = []
        for key, l in top:
            since = next((r['t'] for r in rows if r['io_pairs'].get(key, 0) >= 500), None)
            a_, b_ = (int(x) for x in key.split('-'))
            since_txt = f'{since:.0f}' if since is not None else '- (< 500 km at every listed time)'
            txt.append(f"{key} {l/1e3:.1f}k since {since_txt} (cf {pr[a_]['cf']:.2f}/{pr[b_]['cf']:.2f}, "
                       f"{pr[a_]['nnr_cm_yr']:.1f}/{pr[b_]['nnr_cm_yr']:.1f} cm/yr)")
        start_pairs = sum(l for k, l in last['io_pairs'].items() if all(int(x) < 12 for x in k.split('-')))
        print(f"  {Path(sd).name}: total {last['io_km']/1e3:.1f}k; between two start plates {start_pairs/1e3:.1f}k; " + '; '.join(txt))
    if a.json:
        Path(a.json).write_text(json.dumps({'earth': e, 'node_factor': fnode, 'runs': allr}, indent=1, default=float))
    return allr


if __name__ == '__main__':
    main()
