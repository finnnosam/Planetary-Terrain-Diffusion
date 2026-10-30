"""Where the ocean-floor sediment lies and where it lands, planet against GlobSed (item B-c probe; plan doc, thirty-sixth
pass). Reads saved snapshots only; changes nothing in the model.

  python -m scripts.ocean_sediment_probe outputs\\planet-calibration\\sweep-20261025-170555\\boundary_normal-ring_extension-False_ext_onset_thick_m-45000

Ocean floor = label-free ocean (crust + sediment < 20 km, scripts.crust_review; Earth: CRUST1.0 crsthk < 20 km in the
node's 1 degree cell, and GlobSed valid over more than half the node). Each node is binned by its graph distance to
the nearest subduction boundary (planet: boundary_type 2 nodes; Earth: Bird 2003 SUB steps, every 10 km, on their
nearest nodes) and to the nearest continental node (label-free), and (planet only) by ocean-floor age.

1. Stock: area share, mean sediment and share of the ocean's sediment in each bin, at 0 Myr, against GlobSed
   (scripts.earth_sediment). GlobSed is bulk thickness; the planet's is read as CRUST1.0 and GlobSed count it
   (crust_review.sediment_bulk_m: the compacting column's bulk height with sediment_compaction, else the layer of
   rho_sediment the model carries, ~15% porous): compare the shapes of the profiles (mean in a bin / ocean mean) as
   well as the levels. Arrivals (2) are solid volume (sediment_m) either way.
2. Arrival (the probe): over the last 40 Myr (1 Myr snapshots), the sediment change of every oceanic node between two
   snapshots, following its plate (the node's position one step earlier from the two plate rotations, the nearest node
   there owned by the same plate); new crust and nodes that changed plate are skipped. Positive change = arrival
   (fill-and-spill turbidites and margin deposits; pelagic_m_myr is 0 in the runs read), negative = erosion of floor
   sediment. Checked against the run's own ledger (ocean_sediment_m3 over the same steps). The gross arrival carries
   resampling noise of both signs (nearest-node sampling across sediment gradients); read the net.
3. Four classes by the two distances (200 km): active-margin foot, passive-margin foot, intra-oceanic trench zone,
   open ocean: stock (Earth and planet) and net arrival (planet).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from planet_layer import fields
from planet_layer import graph_landscape as gl
from planet_layer.sphere_graph import CubeSphere
from scripts import earth_sediment
from scripts.crust_review import column_thickness_m, label_free_continent, run_params, sediment_bulk_m
from scripts.margin_width import crust1_cell
from scripts.trench_origin import read_dbf, _xyz

ROOT = Path(__file__).resolve().parents[1]
D_EDGES_KM = (0., 100., 200., 400., 800., 1600., 1e9)
AGE_EDGES = (0., 20., 40., 80., 120., 1e9)
LAST = 40
NEAR_KM = 200.
CROSS = (('active margin foot (continent and trench < 200 km)', True, True),
         ('passive margin foot (continent < 200, trench >= 200 km)', True, False),
         ('intra-oceanic trench zone (continent >= 200, trench < 200 km)', False, True),
         ('open ocean (both >= 200 km)', False, False))


def cross(a, v, dt_, dc_, w=None):
    """Area share, mean of v and share of the integral of v (or w) in the four CROSS classes."""
    rows = []
    tot = (a*(v if w is None else w)).sum()
    for name, nc, nt in CROSS:
        m = ((dc_ < NEAR_KM) == nc) & ((dt_ < NEAR_KM) == nt)
        ww = a[m]*(v[m] if w is None else w[m])
        rows.append({'class': name, 'area_share': float(a[m].sum()/a.sum()),
                     'mean': float(np.average(v[m], weights=a[m])) if m.any() else float('nan'), 'share': float(ww.sum()/tot)})
    return rows


def print_cross(rows, unit='m'):
    for r in rows:
        print(f"    {r['class']:62s} area {r['area_share']:.2f}  mean {r['mean']:7.1f} {unit}  share {r['share']:.2f}")


def bins_label(edges):
    return [f'{edges[i]:.0f}-{edges[i+1]:.0f}' if edges[i+1] < 1e8 else f'>{edges[i]:.0f}' for i in range(len(edges)-1)]


def profile(a, v, x, edges, w=None):
    """Per bin of x: area share, mean of v (area weights), share of the area-integral of v (or of w if given)."""
    tot_a = a.sum(); tot_v = (a*(v if w is None else w)).sum()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        ww = a[m]*(v[m] if w is None else w[m])
        rows.append({'area_share': float(a[m].sum()/tot_a), 'mean': float(np.average(v[m], weights=a[m])) if m.any() else float('nan'),
                     'share': float(ww.sum()/tot_v) if tot_v else float('nan')})
    return rows


def earth_fields(g):
    sed, valid = earth_sediment.load(g.n)
    T = np.loadtxt(ROOT/'data'/'crust1'/'crsthk.xyz')[:, 2]*1e3
    cell = crust1_cell(g)
    cont = label_free_continent(T[cell])
    s = read_dbf(ROOT/'data'/'tectonicplates-master'/'PB2002_steps.dbf')
    sub = s['STEPCLASS'] == 'SUB'
    a = _xyz(s['STARTLONG'][sub], s['STARTLAT'][sub]); b = _xyz(s['FINALLONG'][sub], s['FINALLAT'][sub])
    pts = []
    for p, q, L in zip(a, b, s['STEPLENGTH'][sub]):
        k = max(2, int(np.ceil(L/10.))+1)
        t = np.linspace(0, 1, k)[:, None]
        r = (1-t)*p+t*q; pts.append(r/np.linalg.norm(r, axis=1, keepdims=True))
    trench = np.zeros(g.size, bool); trench[fields.nearest_nodes(g, np.concatenate(pts))] = True
    ocean = ~cont & (valid > .5)
    return {'sed': sed, 'ocean': ocean, 'cont': cont, 'trench': trench}


def planet_fields(seed_dir, t, g):
    f = Path(seed_dir)/'snapshots'/('t+0000.0.npz' if t == 0 else f't{t:+07.1f}.npz')
    z = np.load(f)
    p = run_params(seed_dir)
    T = column_thickness_m(z, p)
    cont = label_free_continent(T)
    return z, {'sed': sediment_bulk_m(z), 'ocean': ~cont, 'cont': cont, 'compaction': bool(p.get('sediment_compaction')),
               'trench': z['boundary_type'] == 2, 'age': np.nan_to_num(z['ocean_age_myr'].astype(np.float64))}


def distances(g, F):
    return (gl.distance_to(F['trench'], g.nbr, g.dist)/1e3, gl.distance_to(F['cont'], g.nbr, g.dist)/1e3)


def arrivals(seed_dir, g):
    """Sediment arriving on (and eroded from) the ocean floor over the last LAST Myr, per node of the later snapshot,
    following the plates; with the distances and age of the later snapshot. Returns arrays summed over the steps and
    the ledger over the same steps."""
    sd = Path(seed_dir)
    st = json.loads((sd/'stats.json').read_text())['series']
    led = {r['time_myr']: r['erosion_totals_km3']['ocean_sediment_m3'] for r in st}
    acc = []
    probe_net = 0.; probe_pos = 0.; ledger = 0.
    for t in range(-LAST, 0):
        f0 = sd/'snapshots'/f't{t:+07.1f}'; f1 = sd/'snapshots'/('t+0000.0' if t+1 == 0 else f't{t+1:+07.1f}')
        z0 = np.load(str(f0)+'.npz'); z1, F1 = planet_fields(sd, t+1, g)
        P0 = {r['id']: np.asarray(r['Q']) for r in json.loads(Path(str(f0)+'-plates.json').read_text())}
        P1 = {r['id']: np.asarray(r['Q']) for r in json.loads(Path(str(f1)+'-plates.json').read_text())}
        pid0, pid1 = z0['plate_id'], z1['plate_id']
        src = np.full(g.size, -1)
        for k in np.unique(pid1):
            if k not in P0:
                continue
            m = np.flatnonzero(pid1 == k)
            x0 = (g.xyz[m]@P1[k])@P0[k].T
            n0 = fields.nearest_nodes(g, x0)
            ok = pid0[n0] == k
            src[m[ok]] = n0[ok]
        use = (src >= 0) & F1['ocean'] & ~z1['new_crust'].astype(bool) & ~z0['continental'][np.clip(src, 0, None)].astype(bool)
        d = np.zeros(g.size)
        d[use] = z1['sediment_m'][use].astype(np.float64)-z0['sediment_m'][src[use]].astype(np.float64)
        dt_, dc_ = distances(g, F1)
        acc.append((d, dt_, dc_, F1['age'], use))
        probe_net += float((d*g.area).sum()); probe_pos += float((np.maximum(d, 0)*g.area).sum())
        ledger += (led[float(t+1)]-led[float(t)])*1e9
    return acc, {'probe_net_km3_yr': probe_net/1e9/LAST/1e6, 'probe_arrival_km3_yr': probe_pos/1e9/LAST/1e6,
                 'ledger_ocean_sediment_km3_yr': ledger/1e9/LAST/1e6}


def seed_dirs(dirs):
    out = []
    for d in dirs:
        d = Path(d)
        out += sorted(d.glob('seed*')) if not (d/'snapshots').exists() else [d]
    return out


def fmt(rows, key, f='{:.2f}'):
    return ' '.join(f.format(r[key]).rjust(7) for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', help='combination folders (all seeds) or seed folders')
    ap.add_argument('--no-arrivals', action='store_true', help='stock only (fast)')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    g = CubeSphere(217)
    out = {}
    E = earth_fields(g)
    edt, edc = distances(g, E)
    o = E['ocean']; ar = g.area[o]
    ocean_mean = np.average(E['sed'][o], weights=ar)
    out['earth'] = {'ocean_area_share': float(ar.sum()/g.area.sum()), 'ocean_mean_m': float(ocean_mean),
                    'by_trench_km': profile(ar, E['sed'][o], edt[o], D_EDGES_KM),
                    'by_continent_km': profile(ar, E['sed'][o], edc[o], D_EDGES_KM),
                    'cross': cross(ar, E['sed'][o], edt[o], edc[o])}
    lab = bins_label(D_EDGES_KM)
    print(f"Earth (GlobSed v3 bulk thickness on label-free ocean, n = 217): ocean {out['earth']['ocean_area_share']:.3f} of the"
          f" surface, mean {ocean_mean:.0f} m")
    print('  distance (km)          ' + ' '.join(s.rjust(7) for s in lab))
    for name in ('by_trench_km', 'by_continent_km'):
        r = out['earth'][name]
        print(f'  {name:15s} area     ' + fmt(r, 'area_share'))
        print(f'  {"":15s} mean m   ' + fmt(r, 'mean', '{:.0f}'))
        print(f'  {"":15s} /mean    ' + ' '.join(f"{x['mean']/ocean_mean:7.2f}" for x in r))
    print_cross(out['earth']['cross'])
    for sd in seed_dirs(a.dirs):
        z, F = planet_fields(sd, 0, g)
        dt_, dc_ = distances(g, F)
        o = F['ocean']; ar = g.area[o]; mean = np.average(F['sed'][o], weights=ar)
        res = {'ocean_area_share': float(ar.sum()/g.area.sum()), 'ocean_mean_m': float(mean),
               'by_trench_km': profile(ar, F['sed'][o], dt_[o], D_EDGES_KM),
               'by_continent_km': profile(ar, F['sed'][o], dc_[o], D_EDGES_KM),
               'by_age_myr': profile(ar, F['sed'][o], F['age'][o], AGE_EDGES),
               'cross': cross(ar, F['sed'][o], dt_[o], dc_[o])}
        kind = 'bulk, compacting column' if F['compaction'] else f"a layer of rho_sediment {run_params(sd)['rho_sediment']:.0f}"
        print(f"\n{sd}\n  stock at 0 Myr: ocean {res['ocean_area_share']:.3f} of the surface, mean {mean:.0f} m ({kind})")
        for name, edges in (('by_trench_km', D_EDGES_KM), ('by_continent_km', D_EDGES_KM), ('by_age_myr', AGE_EDGES)):
            r = res[name]
            print(f'  {name:15s} bins     ' + ' '.join(s.rjust(7) for s in bins_label(edges)))
            print(f'  {"":15s} area     ' + fmt(r, 'area_share'))
            print(f'  {"":15s} mean m   ' + fmt(r, 'mean', '{:.0f}'))
            print(f'  {"":15s} /mean    ' + ' '.join(f"{x['mean']/mean:7.2f}" for x in r))
        print_cross(res['cross'])
        if not a.no_arrivals:
            acc, chk = arrivals(sd, g)
            res['arrival_check'] = chk
            print(f"  arrivals over the last {LAST} Myr: probe net {chk['probe_net_km3_yr']:.2f} km^3/yr (arrival"
                  f" {chk['probe_arrival_km3_yr']:.2f}), ledger ocean_sediment {chk['ledger_ocean_sediment_km3_yr']:.2f}")
            for name, k, edges in (('by_trench_km', 1, D_EDGES_KM), ('by_continent_km', 2, D_EDGES_KM), ('by_age_myr', 3, AGE_EDGES)):
                A = np.zeros(len(edges)-1); P = np.zeros(len(edges)-1); N = np.zeros(len(edges)-1)
                for step in acc:
                    d, x, use = step[0], step[k], step[4]
                    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                        m = use & (x >= lo) & (x < hi)
                        A[i] += g.area[m].sum(); P[i] += (np.maximum(d[m], 0)*g.area[m]).sum(); N[i] += (d[m]*g.area[m]).sum()
                res[f'arrival_{name}'] = {'area_share': (A/A.sum()).tolist(), 'arrival_share': (P/P.sum()).tolist(),
                                          'arrival_m_myr': (P/np.maximum(A, 1)).tolist(), 'net_m_myr': (N/np.maximum(A, 1)).tolist()}
                r = res[f'arrival_{name}']
                print(f'  arrival {name:15s} ' + ' '.join(s.rjust(7) for s in bins_label(edges)))
                print(f'  {"":23s} area ' + ' '.join(f'{x:7.2f}' for x in r['area_share']))
                print(f'  {"":23s} arr. ' + ' '.join(f'{x:7.2f}' for x in r['arrival_share']))
                print(f'  {"":23s} m/Myr' + ' '.join(f'{x:7.1f}' for x in r['arrival_m_myr']))
                print(f'  {"":23s} net  ' + ' '.join(f'{x:7.1f}' for x in r['net_m_myr']))
            Ac = np.zeros(4); Nc = np.zeros(4)
            for d, x_t, x_c, _, use in acc:
                for i, (_, nc, nt) in enumerate(CROSS):
                    m = use & ((x_c < NEAR_KM) == nc) & ((x_t < NEAR_KM) == nt)
                    Ac[i] += g.area[m].sum(); Nc[i] += (d[m]*g.area[m]).sum()
            res['arrival_cross'] = [{'class': c[0], 'area_share': float(Ac[i]/Ac.sum()), 'mean': float(Nc[i]/max(Ac[i], 1)),
                                     'share': float(Nc[i]/Nc.sum())} for i, c in enumerate(CROSS)]
            print('  net arrival by class (m/Myr; share of the net):')
            print_cross(res['arrival_cross'], 'm/Myr')
        out[str(sd)] = res
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    main()
