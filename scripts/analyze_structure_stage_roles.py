"""Analyse saved stage-role probes without sampling or training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from planet_training.data import atomic_json
from scripts.diagnose_terrain_structure import image, legend
from scripts.probe_structure_stage_roles import BRANCH_TITLES


def summary(rows):
    result = {}
    for case in sorted({r['case'] for r in rows}):
        result[case] = {}
        for branch in BRANCH_TITLES:
            selected = [r for r in rows if r['case'] == case and r['branch'] == branch]
            if not selected:
                continue
            result[case][branch] = {
                'comparisons': len(selected),
                'raw_footprint_drift_median_rms_m': float(np.median([r['raw_preservation']['input_footprint_mean_drift_rms_m'] for r in selected])),
                'raw_land_relief_p90_median_m': float(np.median([r['meso_raw']['land_local_relief_p50_p90_m'][1] for r in selected if r['meso_raw']['land_local_relief_p50_p90_m']])),
                'raw_gentle_land_fraction_median': float(np.median([r['meso_raw']['gentle_fraction_of_land'] for r in selected])),
                'raw_negative_prominence_largest_patch_fraction_median': float(np.median([r['meso_raw']['negative_prominence_objects']['largest_area_fraction_of_patch'] for r in selected])),
                'raw_positive_prominence_largest_patch_fraction_median': float(np.median([r['meso_raw']['positive_prominence_objects']['largest_area_fraction_of_patch'] for r in selected])),
            }
    return result


def paired_effects(folder, records):
    pairs = [('oracle-direct', 'oracle-target-detail'), ('oracle-direct', 'oracle-sampled-detail'),
             ('learned-direct', 'learned-sampled-detail')]
    lookup = {(r['case'], r['intermediate_seed'], r['meso_seed'], r['branch']): r for r in records}
    outputs = []; signatures = defaultdict(set); repeats = {}; verification_count = 0
    for r in records:
        path = folder/f'{r["case"]}-i{r["intermediate_seed"]}-{r["branch"]}-m{r["meso_seed"]}.npz'
        with np.load(path) as d:
            expected = np.sign(d['guide23_metres'][..., 47:65, 47:65])*np.sqrt(np.abs(d['guide23_metres'][..., 47:65, 47:65]))
            np.testing.assert_array_equal(d['meso_input_encoded'], expected.astype(np.float32))
            key = r['case'], r['branch'], hashlib.sha256(d['meso_input_encoded'].tobytes()).hexdigest(), r['meso_seed']
            if key in repeats:
                np.testing.assert_array_equal(repeats[key], d['meso_raw_metres'])
                verification_count += 1
            else:
                repeats[key] = d['meso_raw_metres'].copy()
            signatures[r['case'], r['branch']].add(key[2])
    for (case, i, m, name), r in lookup.items():
        for a, b in pairs:
            if name != a or (case, i, m, b) not in lookup:
                continue
            with np.load(folder/f'{case}-i{i}-{a}-m{m}.npz') as left, np.load(folder/f'{case}-i{i}-{b}-m{m}.npz') as right:
                delta_guide = right['guide23_metres']-left['guide23_metres']
                delta_fine = right['meso_raw_metres']-left['meso_raw_metres']
                outputs.append({'case': case, 'intermediate_seed': i, 'meso_seed': m, 'a': a, 'b': b,
                    'guide_change_central_1198km_rms_m': float(np.sqrt(np.mean(delta_guide[..., 30:82, 30:82]**2))),
                    'guide_change_central_369km_rms_m': float(np.sqrt(np.mean(delta_guide[..., 48:64, 48:64]**2))),
                    'raw_meso_change_central_369km_rms_m': float(np.sqrt(np.mean(delta_fine**2)))})
    return {'pairs': outputs, 'unique_guide_counts': [{'case': c, 'branch': b, 'unique_meso_inputs': len(values)} for (c, b), values in signatures.items()],
        'unique_guide_seed_realizations': len(repeats), 'exact_repeat_checks': verification_count,
        'exact_handoff_checks': len(records), 'limit': 'Repeated oracle/direct controls across intermediate seeds are reproducibility checks, not independent samples. Change RMS describes responsiveness, not terrain quality.'}


def focus(folder, settings, cases, label):
    branches = ['modern23-oracle', 'selected', 'oracle-direct', 'oracle-sampled-detail', 'learned-direct', 'learned-sampled-detail']
    i, m = settings['intermediate_seeds'][0], settings['meso_seeds'][0]
    fig, axes = plt.subplots(len(cases), len(branches), figsize=(19.5, len(cases)*3.35+.6), squeeze=False, constrained_layout=True)
    for row, case in enumerate(cases):
        for col, branch in enumerate(branches):
            ax = axes[row, col]; path = folder/f'{case}-i{i}-{branch}-m{m}.npz'
            if not path.exists():
                ax.axis('off'); ax.text(.5, .5, 'No Earth oracle\nfor generated world', ha='center', va='center', transform=ax.transAxes)
                continue
            with np.load(path) as d:
                z = d['meso_raw_metres'][0, 0]
            ax.imshow(image(z, 2880.), extent=(-184.32, 184.32, -184.32, 184.32), interpolation='nearest')
            ax.set_title(case+'\n'+BRANCH_TITLES[branch], fontsize=8)
            ax.set_xlabel('km'); ax.set_ylabel('km'); ax.tick_params(labelsize=7)
    fig.suptitle(f'{label}: SAME 368.64 km per panel; intermediate seed {i}, independent meso seed {m}\nFixed metre colours; NW 5x shading. Every image is a raw learned meso output. Oracle labels identify supplied Earth information.', fontsize=11)
    legend(fig, axes); fig.savefig(folder/f'{label}-focus.png', dpi=130); plt.close(fig)


def all_seeds(folder, settings, case, meso_seed):
    branches = ['selected', 'oracle-direct', 'oracle-sampled-detail', 'learned-direct', 'learned-sampled-detail']
    branches = [b for b in branches if (folder/f'{case}-i{settings["intermediate_seeds"][0]}-{b}-m{meso_seed}.npz').exists()]
    seeds = settings['intermediate_seeds']
    fig, axes = plt.subplots(len(branches), len(seeds), figsize=(3.25*len(seeds), 3.1*len(branches)+.7), squeeze=False, constrained_layout=True)
    for row, branch in enumerate(branches):
        for col, seed in enumerate(seeds):
            with np.load(folder/f'{case}-i{seed}-{branch}-m{meso_seed}.npz') as d:
                z = d['meso_raw_metres'][0, 0]
            ax = axes[row, col]
            ax.imshow(image(z, 2880.), extent=(-184.32, 184.32, -184.32, 184.32), interpolation='nearest')
            ax.set_title(BRANCH_TITLES[branch]+f'\nUpstream seed {seed}', fontsize=8)
            ax.set_xlabel('km'); ax.set_ylabel('km'); ax.tick_params(labelsize=7)
    fig.suptitle(f'{case}: all upstream seeds, fixed meso seed {meso_seed}\n368.64 km per panel; fixed metre colours, NW 5x shading. Oracle-direct repeats are intentional controls.', fontsize=11)
    legend(fig, axes)
    name = f'{case}-all-seeds-m{meso_seed}.png'
    fig.savefig(folder/name, dpi=115); plt.close(fig)
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    args = parser.parse_args(argv)
    folder = Path(args.folder).resolve()
    report = json.loads((folder/'results.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Wait for the complete probe before aggregating')
    settings = report['settings']; records = report['records']
    aggregate = summary(records); effects = paired_effects(folder, records)
    atomic_json(folder/'analysis.json', {'summary': aggregate, 'paired_effects': effects,
        'limits': 'No automatic realism score. Aggregates summarize local descriptors, not world prevalence or geographically paired reconstruction requirements.'})
    focus(folder, settings, [c for c in ('alps', 'california', 'plains') if c in aggregate], 'earth-diagnostics')
    holdouts = [c for c in ('holdout-quiet', 'holdout-rugged', 'holdout-coast') if c in aggregate]
    if holdouts:
        focus(folder, settings, holdouts, 'formal-holdouts')
    generated = [c for c in ('generated-60', 'generated-200') if c in aggregate]
    if generated:
        focus(folder, settings, generated, 'generated-prefixes')
    seed_images = [all_seeds(folder, settings, case, seed) for case in aggregate for seed in settings['meso_seeds']]
    index_path = folder/'index.html'
    if index_path.exists():
        page = index_path.read_text(encoding='utf-8')
        start, end = '<!-- saved-analysis-start -->', '<!-- saved-analysis-end -->'
        if start in page:
            before, rest = page.split(start, 1)
            page = before+rest.split(end, 1)[1]
        links = ''.join(f'<li><a href="{name}">{name[:-4]}</a></li>' for name in seed_images)
        block = start+'<section><h2>Saved-output analysis</h2><p><a href="assessment.md">Assessment</a> · <a href="measurements.md">Descriptive measurements</a> · '
        block += '<a href="earth-diagnostics-focus.png">Earth focus</a> · <a href="formal-holdouts-focus.png">Holdout focus</a> · <a href="generated-prefixes-focus.png">Generated focus</a></p>'
        block += '<p>All seed comparisons, grouped by terrain and fixed meso seed:</p><ul>'+links+'</ul></section>'+end
        page = page.replace('</body>', block+'</body>')
        index_path.write_text(page, encoding='utf-8')
    lines = ['# Paired stage-role measurements', '',
        'These are descriptive measurements. A lower drift or higher relief is not terrain-quality approval. Oracle structure is supplied Earth information. Repeated controls across intermediate seeds are not independent samples.', '',
        '| Case | Branch | Raw meso footprint drift, median RMS (m) | Local relief p90, median (m) | Gentle land fraction, median |',
        '|---|---|---:|---:|---:|']
    for case, branches in aggregate.items():
        for branch, r in branches.items():
            lines.append(f'| {case} | {branch} | {r["raw_footprint_drift_median_rms_m"]:.1f} | {r["raw_land_relief_p90_median_m"]:.1f} | {r["raw_gentle_land_fraction_median"]:.3f} |')
    lines += ['', f'Exact handoff checks: {effects["exact_handoff_checks"]}. Exact repeated-input/noise output checks: {effects["exact_repeat_checks"]}. Unique guide/noise realizations: {effects["unique_guide_seed_realizations"]}.', '',
        'All seed panels are linked in index.html. Focus panels show the preregistered first seed combination; they are not selected for attractiveness. See analysis.json for all paired changes and unique-input counts.']
    (folder/'measurements.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Analysed {len(records)} saved comparisons; {effects["unique_guide_seed_realizations"]} unique guide/noise realizations; exact handoffs/repeats passed. {folder}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
