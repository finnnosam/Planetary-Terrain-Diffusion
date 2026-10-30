"""Render all saved structure-review seeds and summarize central geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from planet_training.data import atomic_json
from planet_training.terrain_structure import decode, conservative_resample
from scripts.review_intermediate_structure import widths, frame
from planet_training.data import bridge_patch
from toy_meso.sphere import GlobalRaster


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', default='outputs/structure/terrain-40-pilot-v1/review-20261013-182311-823544')
    a = p.parse_args(argv); root = Path(a.review); out = root/'analysis'; out.mkdir(exist_ok=True)
    d = json.loads((root/'regional-review.json').read_text()); seeds = d['settings']['seeds']
    if d['status'] != 'complete': raise ValueError('Review is incomplete')
    from scripts.diagnose_terrain_structure import image, legend, gauss
    import matplotlib.pyplot as plt
    for group in ['earth', 'generated']:
        cases = [c for c in d['settings']['cases'] if c[3] == group]
        fig, axs = plt.subplots(len(cases)*2, len(seeds), figsize=(14, 5*len(cases)), squeeze=False, constrained_layout=True)
        for r, (region, _, _, _) in enumerate(cases):
            for m, branch in enumerate(['control', 'new-stage']):
                for c, seed in enumerate(seeds):
                    z = np.load(root/f'{region}-seed{seed}-{branch}-intermediate.npy')[30:-30, 30:-30]
                    ax = axs[2*r+m, c]; ax.imshow(image(z, 23040.), extent=(-599.04, 599.04, -599.04, 599.04), interpolation='nearest')
                    ax.set_title(f'{region}\n{branch}, seed {seed}', fontsize=9); ax.set_xticks([-400, 0, 400]); ax.set_yticks([-400, 0, 400]); ax.tick_params(labelsize=7)
        fig.suptitle(f'{group.capitalize()}: constrained control and learned stage, all paired seeds\nEvery panel 1198 km square; same metre colours and NW 5x hillshade.')
        legend(fig, axs); fig.savefig(out/f'{group}-all-seeds.png', dpi=115); plt.close(fig)
    earth = GlobalRaster(np.load('outputs/planet_training/data/cache/etopo-filtered.npy', mmap_mode='r'))
    fig, axs = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    for r, (region, lat, lon, _) in enumerate(d['settings']['cases'][:3]):
        target, _ = bridge_patch(earth, frame(lat, lon)); target = decode(torch.from_numpy(target.astype(np.float32)))[30:-30, 30:-30].numpy()
        fields = [('Modern Earth', target)]+[(branch, np.load(root/f'{region}-seed42-{branch}-intermediate.npy')[30:-30, 30:-30]) for branch in ['control', 'new-stage', 'Earth-oracle']]
        for c, (title, z) in enumerate(fields):
            ax = axs[r, c]; ax.imshow(image(z, 23040.), extent=(-599.04, 599.04, -599.04, 599.04), interpolation='nearest')
            ax.set_title(region+'\n'+title, fontsize=10); ax.set_xticks([-400, 0, 400]); ax.set_yticks([-400, 0, 400]); ax.set_xlabel('km')
    fig.suptitle('Known Earth geometry, constrained control, learned stage and Earth-derived reference\nSeed 42; all panels 1198 km square. The reference is not learned generation.')
    legend(fig, axs); fig.savefig(out/'earth-reference-comparison.png', dpi=115); plt.close(fig)
    records = []
    for entry in d['regions']:
        stats = {'region': entry['region'], 'seed': entry['seed'], 'branches': {}}
        for branch in entry['branches']:
            z = np.load(root/f'{entry["region"]}-seed{entry["seed"]}-{branch}-intermediate.npy')[30:-30, 30:-30]
            smooth = gauss(z, 2.); dy, dx = np.gradient(smooth, 23.04)
            stats['branches'][branch] = {'land_mean_m': float(z[z > 0].mean()) if (z > 0).any() else None,
                'above_1000_fraction': float((z > 1000).mean()),
                'WE_above_1000': widths(z[26], 1000, 23.04), 'SN_above_1000': widths(z[::-1, 26], 1000, 23.04),
                'broad_slope_p90_m_per_km': float(np.percentile(np.hypot(dx, dy), 90))}
        records.append(stats)
    count = 0
    for file in root.glob('*.npy'):
        if not np.isfinite(np.load(file)).all(): raise FloatingPointError(f'Nonfinite saved field: {file}')
        count += 1
    atomic_json(out/'central-metrics.json', {'method': 'Central 52 cells at 23.04 km; Gaussian sigma 46.08 km before slopes. Absolute threshold widths are descriptive, not a mountain classifier.', 'finite_arrays_checked': count, 'regions': records})
    print('All-seed figures and measurements saved:', out)
    return 0


if __name__ == '__main__': raise SystemExit(main())
