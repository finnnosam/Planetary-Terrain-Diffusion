"""Eight bounded meso samples isolate the v1 constraint grid and smooth fix."""
import argparse
from pathlib import Path

import numpy as np
import torch

from planet_training.data import atomic_json
from planet_training.terrain_structure import encode, decode, project, constraint_metrics
from planet_training.smooth_conservation import VERSION, project_smooth, grid_jumps
from scripts.review_intermediate_structure import sha, MESO_SHA
from toy_meso.common import load_checkpoint, sample_patches


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', default='outputs/structure/terrain-40-pilot-v1/review-20261013-182311-823544')
    p.add_argument('--out'); p.add_argument('--device', default='cuda')
    a = p.parse_args(argv); source = Path(a.review)
    out = Path(a.out) if a.out else source/'analysis/meso-conservation-probe'
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a fresh probe directory')
    out.mkdir(parents=True)
    checkpoint = source/'meso-checkpoint.pt'
    if sha(checkpoint) != MESO_SHA: raise ValueError('Frozen meso model changed')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, norm, _ = load_checkpoint(checkpoint, a.device)
    from scripts.diagnose_terrain_structure import image, legend
    import matplotlib.pyplot as plt
    records = []
    for region in ['earth-alps', 'earth-california', 'earth-plains', 'new-highland']:
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for row, branch in enumerate(['control', 'new-stage']):
            z = torch.from_numpy(np.load(source/f'{region}-seed42-{branch}-intermediate.npy'))[None, None].to(a.device)
            c = encode(z[..., 47:65, 47:65]); fine, up = sample_patches(model, norm, c, 32, 42, a.device)
            raw, base = decode(fine), decode(up)
            old = project(raw, base, 8, 2880.)
            smooth, info = project_smooth(raw, base, return_info=True)
            saved = np.load(source/f'{region}-seed42-{branch}-meso.npy')
            error = float(np.max(np.abs(old[0, 0].cpu().numpy()-saved)))
            if error > .01: raise ValueError(f'Replayed meso differs from saved review: {error} m')
            record = {'region': region, 'branch': branch, 'seed': 42, 'replay_max_error_m': error,
                      'smooth_info': info, 'variants': {}}
            for col, (label, field) in enumerate([('Raw meso', raw), ('Old block correction', old), ('Smooth correction', smooth)]):
                values = field[0, 0].cpu().numpy()
                np.save(out/f'{region}-{branch}-{label.replace(" ", "-")}.npy', values)
                record['variants'][label] = dict(grid_jumps(field), constraints=constraint_metrics(field, base, 8, 2880.))
                axs[row, col].imshow(image(values, 2880.), extent=(-184.32, 184.32, -184.32, 184.32))
                axs[row, col].set_title(branch+'\n'+label); axs[row, col].set_xlabel('km'); axs[row, col].set_ylabel('km')
            fig.suptitle(region+': same frozen meso weights, guide, seed 42 and 32 steps\nOnly the correction changes. Every panel 369 km square; same colours and NW 5x shading.')
            records.append(record)
        legend(fig, axs); fig.savefig(out/f'{region}.png', dpi=120); plt.close(fig)
        print('Completed bounded conservation probe:', region, flush=True)
    atomic_json(out/'probe-results.json', {'version': VERSION, 'source_review': str(source.resolve()), 'records': records,
        'new_structure_training': False, 'new_structure_sampling': False, 'meso_samples': 8, 'full_planet_generation': False,
        'limits': 'Four seed-42 regions only; fixes postprocessing grid continuity, not learned belt placement. Coarse means are local area-weighted baseline means.'})
    print('Bounded probe complete:', out, flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
