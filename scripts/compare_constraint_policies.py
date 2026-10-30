"""Show what the 200 m lock removes from the structural learning target (no model, a few seconds).

For each Earth case, draw the unconstrained modern 40 km field, the coarse guide baseline,
and the feasible target under each constraint policy, with the frozen-land share and the
arrangement metrics. Writes outputs/structure/plan-v2/constraint-policies.{png,json}.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training import arrangement_metrics as am
from planet_training import structure_plan_v2 as v2
from planet_training.elevation_shading import show, legend
from planet_training.terrain_structure import StructureData, DEFAULT_DATA, decode, conservative_resample
from scripts.review_plan_sources import ROUND_TRIPS, JOINT, NAMED_EARTH, NAMED_HOLDOUTS

ROOT = Path(__file__).resolve().parents[1]


def cases():
    data = StructureData(DEFAULT_DATA)
    for name in NAMED_EARTH:
        t = np.load(ROUND_TRIPS/f'{name}.npz')
        yield name, torch.from_numpy(t['shared_coarse_encoded']), conservative_resample(torch.from_numpy(t['reference23_metres'])[:, 0], 64)[:, None]
    for name, index in NAMED_HOLDOUTS.items():
        yield name, torch.from_numpy(np.load(JOINT/f'{name}-s42-joint.npz')['coarse_encoded']), torch.from_numpy(np.array(data.arrays['val'][index]))[None, None]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', default=str(ROOT/'outputs/structure/plan-v2'))
    out = Path(p.parse_args(argv).out); out.mkdir(parents=True, exist_ok=True)
    rows, items = [], list(cases())
    fig, axs = plt.subplots(len(items), 4, figsize=(15, 3.75*len(items)), squeeze=False, constrained_layout=True)
    for i, (name, coarse, raw) in enumerate(items):
        base = v2.target_with(raw, coarse, 'locked-200')[1]
        show(axs[i, 0], raw[0, 0, 16:48, 16:48].numpy(), f'{name}: modern 40 km (no constraints)', 40320.)
        show(axs[i, 1], base[0, 0, 16:48, 16:48].numpy(), 'coarse guide baseline', 40320.)
        for j, policy in enumerate(v2.POLICIES):
            z = decode(v2.target_with(raw, coarse, policy)[0])
            m = am.arrangement(z.numpy(), base.numpy())
            frozen = float((v2.eligibility_for(base, policy) == 0)[base > 0].float().mean())
            rows.append({'case': name, 'policy': policy, 'frozen_land_fraction': frozen, **{k: m[k] for k in am.METRICS}})
            show(axs[i, 2+j], z[0, 0, 16:48, 16:48].numpy(),
                 f'feasible target, {policy}\nfrozen land {frozen:.0%}; flanked-low {m["flanked_low_fraction"]:.2f}', 40320.)
    fig.suptitle('Feasible 40 km learning targets under each constraint policy; each panel 1,290 km\n'
                 'Sea level 0 m in both; coast-only freezes only ocean cells and keeps land >= 1 m', fontsize=11)
    legend(fig, axs); fig.savefig(out/'constraint-policies.png', dpi=85); plt.close(fig)
    (out/'constraint-policies.json').write_text(json.dumps({'metrics_spec': am.SPEC, 'rows': rows}, indent=1), encoding='utf-8')
    for r in rows:
        print(f"{r['case']:18s} {r['policy']:10s} frozen land {r['frozen_land_fraction']:5.1%}  flanked-low {r['flanked_low_fraction']:.3f}  "
              f"long-ridge {r['long_ridge_fraction']:.2f}  connected-low {r['connected_low_fraction']:.2f}")
    print(f'Wrote {out/"constraint-policies.png"}')


if __name__ == '__main__':
    main()
