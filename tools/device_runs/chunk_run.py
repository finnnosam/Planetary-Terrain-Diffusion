"""Chunked n = 217 run with simulate_plates.run_seed's outputs, resumable across calls (device shell: each call <= 180 s,
background jobs are stopped). The run is pickled to ~/work/pk between calls (tools/device_runs/simpickle.py; ~250 MB;
writing it into the mounted repo is too slow). Two runs in parallel do ~45-50 steps per 140 s call (2 cores).

  PYTHONPATH=$PWD:$PWD/tools/device_runs python3 tools/device_runs/chunk_run.py <combo folder> <seed> '<params json>' 140

Repeat the call until it prints 'done'. Trajectories differ from the Windows runs (pre-checks only)."""
import json, sys, time
from pathlib import Path
import numpy as np
import simpickle
from planet_layer import fields
from planet_layer.sphere_graph import CubeSphere
from planet_layer.plates import PlateSim
from scripts import simulate_plates as SP

out, seed, params, budget = Path(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3]), float(sys.argv[4])
t0 = time.perf_counter()
sdir = out/f'seed{seed}'; sdir.mkdir(parents=True, exist_ok=True)
pk = Path.home()/'work'/'pk'/(out.name+f'-seed{seed}.pkl'); pk.parent.mkdir(exist_ok=True)
YEARS, SAVE_LAST, SAVE_EVERY = 250., 40., 1

def record(s, series, maps):
    st = s.state()
    series.append(SP.summary(s, st))
    late = s.t > -SAVE_LAST-1e-6
    if (late and round(s.t) % SAVE_EVERY == 0) or round(s.t) % 10 == 0:
        SP.save_snapshot(sdir/'snapshots', s, st)
    if round(s.t) % 25 == 0:
        maps.append((s.t, st['elevation_m'], st['boundary_type']))

if pk.exists():
    sim, series, maps, secs = simpickle.load(pk)
else:
    g = CubeSphere(217)
    sim = PlateSim(g, seed=seed, t0=-YEARS, params=params)
    series, maps, secs = [], [], 0.
    record(sim, series, maps)
while sim.t < -1e-6 and time.perf_counter()-t0 < budget:
    sim.step()
    record(sim, series, maps)
secs += time.perf_counter()-t0
if sim.t < -1e-6:
    simpickle.save((sim, series, maps, secs), pk)
    print(f'seed {seed} t={sim.t:.0f} plates {len(sim.plates)}', flush=True)
    sys.exit(0)
st = sim.state()
try:
    SP.draw(sdir, maps, st, fields.equirect_nodes(sim.g, 360))
except Exception as e:
    print('draw skipped:', e)
v = [r['continental_volume_km3'] for r in series]
vt = [r['continental_volume_km3']+(r['continental_sediment_km3'] if sim.p['continental_sediment'] else 0.) for r in series]
eff = {k: v_ for k, v_ in sim.p.items() if isinstance(v_, (bool, int, float, str, list, tuple, type(None)))}
result = {'seed': seed, 'n': 217, 'years': YEARS, 'params': params, 'params_effective': eff, 'seconds': secs,
          'present': series[-1], 'continental_volume_change': vt[-1]/vt[0]-1, 'continental_crust_volume_change': v[-1]/v[0]-1,
          'max_pinholes': max(r['pinholes'] for r in series), 'earth': SP.EARTH, 'series': series,
          'event_list': [{k: (float(e[k]) if isinstance(e[k], (float, np.floating)) else e[k])
                          for k in ('t', 'kind', 'plate', 'new_plate', 'merged', 'area_mkm2') if k in e} for e in sim.events]}
(sdir/'stats.json').write_text(json.dumps(result, indent=1, default=float))
pk.unlink()
print(f'seed {seed} done, {secs:.0f} s, plates {len(sim.plates)}', flush=True)
