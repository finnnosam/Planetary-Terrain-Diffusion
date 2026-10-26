"""Log what ends extension events (thirty-third pass): every merge that touches a running event, by role (child /
parent / third plate, with_partner = the halves welded or docked onto each other), the start records, and each half's
boundary types with third plates while the event runs. The sweep baseline of the E3 / E4 sweeps; one seed, n = 217.

  python tools/extension_event_log.py free 125 outputs/e4-log.json [ring|node] [seed] [onset_m]

(ext_side, Myr from -250, output JSON; boundary_normal default node, seed 1001, onset 50000). ~40 min for 125 Myr on
2 cores. Then read: merges with role_small 'child' and role_big 'parent' after exactly merge_after_myr = self-docking."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from planet_layer import plates as P
from planet_layer.sphere_graph import CubeSphere
side, years, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
bn = sys.argv[4] if len(sys.argv) > 4 else 'node'
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1001
onset = float(sys.argv[6]) if len(sys.argv) > 6 else 50000.
prm = {'erosion_K_factor': 1.0, 'erosion_theta': 200., 'erosion_Sc': .036, 'margin_shortening': True, 'sediment_accretion': .13,
       'continental_sediment': True, 'crust_kappa_normal_m2_yr': 30., 'ext_held_rule': 'direction', 'ext_hot_myr': 100.,
       'ext_clock_flow': True, 'ext_pair_motion': 'union', 'extension': True, 'ext_onset_thick_m': onset, 'ext_cut': 'band',
       'ext_side': side, 'boundary_normal': bn}
sim = P.PlateSim(CubeSphere(217), seed=seed, t0=-250., params=prm)
log = {'merges': [], 'lost': [], 'track': [], 'starts': [], 'ends': []}
orig_merge = sim.merge
def merge(big, small):
    pairs = {(ev['parent'], ev['child']): ev for ev in sim.ext_events}
    rec = {'t': sim.t, 'big': int(big.id), 'small': int(small.id)}
    for (a, b), ev in pairs.items():
        if {big.id, small.id} & {a, b}:
            rec.update({'event_t0': ev['t0'], 'parent': a, 'child': b,
                        'role_small': 'child' if small.id == b else 'parent' if small.id == a else 'third',
                        'role_big': 'child' if big.id == b else 'parent' if big.id == a else 'third',
                        'with_partner': {big.id, small.id} == {a, b}})
    log['merges'].append(rec)
    return orig_merge(big, small)
sim.merge = merge
g = sim.g
nb = g.nbr; okn = nb >= 0
NAMES = {0: 'none', P.RIDGE: 'ridge', P.SUBDUCTION: 'subduction', P.COLLISION: 'collision', P.TRANSFORM: 'transform'}
nstart = 0
while sim.t < -250.+years-1e-6:
    sim.step()
    st = [e for e in sim.events if e['kind'] == 'extension']
    for e in st[nstart:]:
        log['starts'].append({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in e.items()
                              if k in ('t', 'plate', 'new_plate', 'far_collision', 'far_collision_sides', 'moved_larger', 'piece_mkm2', 'L', 'W', 'v', 'across')})
    nstart = len(st)
    on = sim.owner[np.where(okn, nb, 0)]
    for ev in sim.ext_events:
        rec = {'t': sim.t, 't0': ev['t0'], 'child': ev['child'], 'parent': ev['parent']}
        for role, pid in (('child', ev['child']), ('parent', ev['parent'])):
            other = ev['parent'] if role == 'child' else ev['child']
            mine = sim.owner == pid
            edge = mine & np.any(okn & (on != pid) & (on != other), axis=1)
            bt = sim.btype[edge]
            rec[role] = {NAMES.get(int(k), str(k)): int((bt == k).sum()) for k in np.unique(bt)}
            # the third plates it touches, by node count
            nbo = np.where(okn[edge], on[edge], -1)
            tp, cn = np.unique(nbo[(nbo != pid) & (nbo != other) & (nbo >= 0)], return_counts=True)
            rec[role+'_third'] = {int(a): int(c) for a, c in zip(tp, cn)}
            rec[role+'_area'] = float(g.area[mine].sum()/1e12)
        log['track'].append(rec)
    log['lost'] = [e for e in sim.events if e['kind'] == 'extension lost']
    log['ends'] = [{k: e[k] for k in ('t', 'myr', 'zone_thick_km', 'guard')} for e in sim.events if e['kind'] == 'extension end']
    if int(round(sim.t)) % 10 == 0:
        json.dump(log, open(out, 'w'), default=float)
        print(sim.t, len(log['starts']), len(log['lost']), flush=True)
json.dump(log, open(out, 'w'), default=float)
print('done', flush=True)
