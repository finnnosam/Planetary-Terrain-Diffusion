"""Cheap Earth-reference test of 40-km representation and its 23-km handoff."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from planet_training.adaptation import STRUCTURE_RECIPE, coarse_context, degrade_guides
from planet_training.data import atomic_json, bridge_patch
from planet_training.terrain_structure import (CONSERVE_FACTOR, HANDOFF_FACTOR, OBJECTIVE,
    conservative_resample, decode, target_for, handoff, constraint_metrics)
from scripts.review_intermediate_structure import CASES, frame
from toy_meso.model import upsample_condition
from toy_meso.sphere import GlobalRaster


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='outputs/structure/checks/representation-v1')
    a = p.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    earth = GlobalRaster(np.load('outputs/planet_training/data/cache/etopo-filtered.npy', mmap_mode='r'))
    from scripts.diagnose_terrain_structure import image, legend
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    records = []
    for row, (label, lat, lon, _) in enumerate(CASES[:3]):
        target, coarse = bridge_patch(earth, frame(lat, lon))
        context = coarse_context(earth, frame(lat, lon), 112)
        c = torch.from_numpy(coarse)[None, None]; ctx = torch.from_numpy(context)[None, None]
        c = degrade_guides(ctx, c, torch.Generator().manual_seed(42), sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
        true23 = decode(torch.from_numpy(target.astype(np.float32))[None, None])
        raw40 = conservative_resample(true23[:, 0], 64)[:, None]
        encoded40, base40 = target_for(raw40, c); oracle40 = decode(encoded40)
        final, _, base23 = handoff(oracle40, base40, upsample_condition(c, 7), c)
        checks = {'structure': constraint_metrics(oracle40, base40, CONSERVE_FACTOR, 40320.),
                  'handoff': constraint_metrics(final, base23, HANDOFF_FACTOR, 23040.)}
        for values in checks.values():
            if values['sign_disagreement_fraction'] or values['protected_height_change_max_m'] > .005 or values['added_footprint_mean_drift_max_m'] > .05:
                raise ValueError(f'Representation constraints failed: {checks}')
        records.append({'region': label, 'latitude': lat, 'longitude': lon, 'checks': checks})
        for name, z in [('true23', true23), ('base23', base23), ('oracle40', oracle40), ('handoff23', final)]:
            np.save(out/f'{label}-{name}.npy', z[0, 0].numpy())
        fields = [('Modern 23-km averages', true23), ('Smoothed input guide', base23),
                  ('Constrained Earth 40-km reference', F.interpolate(oracle40, size=(112, 112), mode='bilinear', align_corners=False)),
                  ('Reference handed to 23 km', final)]
        for col, (name, z) in enumerate(fields):
            ax = axs[row, col]
            ax.imshow(image(z[0, 0, 30:-30, 30:-30].numpy(), 23040.), extent=(-599.04, 599.04, -599.04, 599.04), interpolation='nearest')
            ax.set_title(label+'\n'+name, fontsize=9); ax.set_xlabel('km'); ax.set_xticks([-400, 0, 400]); ax.set_yticks([-400, 0, 400])
    fig.suptitle('Earth-reference representation and handoff check; no learned model\nAll panels 1198 km square; same elevation colours and NW 5x shading. 645-km means and shallow input elevations protected.')
    legend(fig, axs); fig.savefig(out/'reference-handoff.png', dpi=115); plt.close(fig)
    atomic_json(out/'checks.json', {'status': 'passed', 'objective': OBJECTIVE, 'regions': records,
        'model_sampling': False, 'training': False, 'full_planet_generation': False,
        'limits': 'Known Earth terrain only; representability and registration checks are not evidence of learned generation quality. 40-km display column is interpolated to the same 23-km display grid.'})
    print(f'Representation checks passed: {out}. No model inference or training.', flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
