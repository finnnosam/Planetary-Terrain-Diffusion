"""Rebuild the trench-flux band terms (plates._subduction_thickening, trench_flux 'band') from saved snapshots and compare
them with the contour-typed consumption (B-d2, plan doc fortieth pass). Changes nothing in the model.

  python tools/arc_growth_probe.py outputs/planet-calibration/sweep-20261027-210209/arc_H_ocean_m-1000_arc_accretion-False/seed1001

Per snapshot (0 Myr): the oceanic band term's volume at H = 1000 m (add_km3_yr) and add / H against the contour's
intra-oceanic consumption (ocean_band_over_contour_io_consumption), the continental and subduction-erosion bands against
the contour's consumption, which contour class the band volume sits behind (nearest segment), node vs contour lengths.
The trench flag is rebuilt from the node rule (score: continental first, younger ocean overrides)."""
import json, sys
from pathlib import Path
import numpy as np
from planet_layer import plates as P
from planet_layer.sphere_graph import CubeSphere

g = CubeSphere(217)
nbr, dist, area = g.nbr, g.dist, g.area
H, d0, sg = 1000., 150e3, 50e3

def load(sd, t):
    f = Path(sd)/'snapshots'/('t+0000.0.npz' if t == 0 else f't{t:+07.1f}.npz')
    z = np.load(f)
    pj = json.loads(Path(str(f)[:-4]+'-plates.json').read_text())
    return z, pj

def probe(sd, t):
    z, pj = load(sd, t)
    own = z['plate_id'].astype(np.int64); cont = z['continental'].astype(bool)
    bt = z['boundary_type']; rate = z['convergence_mm_yr'].astype(np.float64)*1e-3
    age = z['crust_age_myr'].astype(np.float64)
    share = {p['id']: p['continental_km2']/510.1e6 for p in pj}
    sh = np.array([share.get(int(o), 0.) for o in own])
    score = np.where(cont, 1e6+1e5*sh, -age)
    sub = np.flatnonzero(bt == 2)
    trench = np.zeros(g.size, bool); other = np.full(g.size, -1)
    for i in sub:                       # partner: the other-plate neighbour typed subduction (first found)
        js = [j for j in nbr[i] if j >= 0 and own[j] != own[i]]
        if not js: continue
        jj = [j for j in js if bt[j] == 2] or js
        j = jj[0]; other[i] = j
        trench[i] = score[i] < score[j]
    src = (bt == 2) & ~trench
    lab = np.where(src, np.arange(g.size), -1).astype(np.int64)
    near, d = P.nearest_label(lab, nbr, dist)
    ok = (near >= 0) & (d < 1000e3); ok[ok] &= own[ok] == own[near[ok]]
    v = np.clip(rate[np.clip(near, 0, None)], 0, None)
    shape = np.exp(-.5*((d-d0)/sg)**2)/(sg*np.sqrt(2*np.pi))
    m = ok & ~cont
    add = np.where(m, v*H*shape*area, 0.)              # m^3/yr
    # per source node: band volume / (v H) = effective band length; line length per node ~ sqrt(area)
    S = np.flatnonzero(src)
    band = np.bincount(near[m], weights=(shape*area)[m], minlength=g.size)
    bandall = np.bincount(near[ok], weights=(shape*area)[ok], minlength=g.size)
    ell = np.sqrt(area)
    both = src & (other >= 0) & src[np.clip(other, 0, None)]   # partner also overriding
    srcc = src & cont
    vS = np.clip(rate[S], 0, None)
    line_exp = (vS*H*ell[S]).sum()
    out = {'t': t, 'add_km3_yr': add.sum()/1e9, 'line_exp_km3_yr': line_exp/1e9,
           'src_nodes': int(src.sum()), 'src_cont_share': float(srcc[S].mean()), 'both_overriding_share': float(both[S].mean()),
           'add_from_cont_src': float(add[m & srcc[np.clip(near,0,None)]].sum()/max(add.sum(),1)),
           'add_from_both': float(add[m & both[np.clip(near,0,None)]].sum()/max(add.sum(),1)),
           'band_len_ratio_ocean_src': float(bandall[S[~cont[S]]].sum()/ell[S[~cont[S]]].sum()),
           'band_ocean_ratio_cont_src': float(band[S[cont[S]]].sum()/ell[S[cont[S]]].sum()),
           'band_ocean_ratio_ocean_src': float(band[S[~cont[S]]].sum()/ell[S[~cont[S]]].sum())}
    from scripts.boundary_lengths import typed_segments
    ts = typed_segments(g, own, pj, cont)
    L, typ, conv, na, nb = ts['L'], ts['typ'], ts['conv'], ts['na'], ts['nb']
    oo = ~(cont[na] | cont[nb]); s2 = typ == 2
    out['contour_io_km'] = float(L[s2 & oo].sum()); out['contour_cont_km'] = float(L[s2 & ~oo].sum())
    out['contour_io_consumption_km2_yr'] = float((L*conv)[s2 & oo].sum()*1e-6)
    out['contour_all_consumption_km2_yr'] = float((L*conv)[s2].sum()*1e-6)
    So = S[~cont[S]]
    out['node_line_ocean_src_km'] = float(ell[So].sum()/1e3)
    out['node_consumption_ocean_src_km2_yr'] = float((np.clip(rate[So], 0, None)*ell[So]).sum()/1e6)
    out['effective_consumption_add_over_H'] = out['add_km3_yr']*1e9/H/1e6
    mc = ok & cont
    out['cont_band_over_contour_cont_consumption'] = float((v*shape*area)[mc].sum()/1e6/max((L*conv)[s2 & ~oo].sum()*1e-6, 1e-9))
    shp_se = np.exp(-.5*((d-50e3)/40e3)**2)/(40e3*np.sqrt(2*np.pi))
    out['erosion_band_over_all_consumption'] = float((v*shp_se*area)[ok].sum()/1e6/max((L*conv)[s2].sum()*1e-6, 1e-9))
    out['ocean_band_over_contour_io_consumption'] = out['effective_consumption_add_over_H']/out['contour_io_consumption_km2_yr']
    # contour class at each source node: the class of the contour segments whose side nodes (na / nb) are this node
    cls = np.full(g.size, -1)        # -1 none; 0 io subduction; 1 continental subduction; 2 transform; 3 ridge; 4 collision; 5 untyped
    code = np.where(s2 & oo, 0, np.where(s2, 1, np.where(typ == 4, 2, np.where(typ == 1, 3, np.where(typ == 3, 4, 5)))))
    pri = np.array([0, 1, 4, 2, 3, 5])   # priority when a node has several: io > cont > collision > transform > ridge > untyped
    best = np.full(g.size, 99)
    from planet_layer import fields
    mn = fields.nearest_nodes(g, ts['mid'])
    np.minimum.at(best, mn, pri[code])
    inv = np.argsort(pri)
    cls = np.where(best < 99, inv[np.clip(best, 0, 5)], -1)
    seedlab = np.where(cls >= 0, cls, -1).astype(np.int64)
    cls, dcls = P.nearest_label(seedlab, nbr, dist)
    out['src_dist_to_segment_km_p50_p90'] = [round(float(np.percentile(dcls[S], q))/1e3, 1) for q in (50, 90)]
    srcn = np.clip(near, 0, None)
    tot = max(add.sum(), 1.)
    names = {-1: 'no segment', 0: 'io sub', 1: 'cont sub', 2: 'transform', 3: 'ridge', 4: 'collision', 5: 'untyped'}
    out['add_by_contour_class'] = {names[c]: round(float(add[m & (cls[srcn] == c)].sum()/tot), 3) for c in names}
    So2 = S[~cont[S]]
    out['ocean_src_nodes_by_class'] = {names[c]: round(float((cls[So2] == c).mean()), 3) for c in names}
    return out

if __name__ == '__main__':
    for sd in sys.argv[1:]:
        for t in (0.,):
            print(sd.split('/')[-2][-22:], Path(sd).name, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in probe(sd, t).items()}), flush=True)
