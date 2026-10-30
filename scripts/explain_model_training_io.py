"""Illustrate saved training records and inexpensive fresh samples; never train."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_terrain_structure import image, CMAP, NORM
from planet_training.adaptation import paired_degradation, degrade_guides, RECIPE, MESO_RECIPE, STRUCTURE_RECIPE
from planet_training.terrain_structure import decode, encode, eligibility
from toy_meso.common import load_checkpoint, sample_patches, pick_device

BRIDGE_IDS = [3593, 2751, 4682]
MESO_IDS = [31, 9298, 2536]
LABELS = ['Mountainous interior', 'Lowland interior', 'Mountainous coast']
NOISE_NORM = Normalize(-2, 2)
MASK_CMAP = ListedColormap(['#16465e', '#a6af70'])


def array(x):
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_records(folder, key, ids, device):
    a = np.load(folder/f'train_{key}.npy', mmap_mode='r')
    return torch.from_numpy(np.array(a[ids], dtype=np.float32))[:, None].to(device)


def panel(ax, z, kind, spacing, title=''):
    z = np.asarray(z)
    extent = z.shape[-1]*spacing/1000/2
    opts = dict(extent=[-extent, extent, -extent, extent], interpolation='nearest')
    if kind == 'elevation': ax.imshow(image(z, spacing), **opts)
    elif kind == 'noise': ax.imshow(z, cmap='RdBu_r', norm=NOISE_NORM, **opts)
    elif kind == 'mask': ax.imshow(z, cmap=MASK_CMAP, vmin=0, vmax=1, **opts)
    elif kind == 'eligibility': ax.imshow(z, cmap='Greys_r', vmin=0, vmax=1, **opts)
    if title: ax.set_title(title, fontsize=10)
    ax.set_xticks([-extent, 0, extent], [f'{-extent:g}', '0', f'{extent:g}'], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel('Local distance (km)', fontsize=8)


def bars(fig, noise=False, coast=False, global_view=False):
    y = .12 if global_view else .065
    cax = fig.add_axes([.13, y, .57, .016 if global_view else .012])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 2) if coast else NORM,
        cmap=MASK_CMAP if coast else CMAP), cax=cax, orientation='horizontal')
    if coast:
        cb.set_ticks([.5, 1.5], labels=['Ocean', 'Land'])
        cb.set_label('Land/ocean classes; no elevation is predicted by this stage.', fontsize=9)
    else:
        cb.set_ticks([-6000, -3000, 0, 1000, 2000, 4000, 6000])
        cb.set_label('Magnitude / signed elevation (m); unshaded world display.' if global_view else
            'Elevation (m); colour clipping only. Same NW light, 5x shading throughout.', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if noise:
        cax = fig.add_axes([.77, y, .17, .016 if global_view else .012])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NOISE_NORM, cmap='RdBu_r'), cax=cax, orientation='horizontal')
        cb.set_label('Noisy field: model units', fontsize=9)
        cb.ax.tick_params(labelsize=8)


def regional_figure(out, name, title, data, spacing, note):
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    fig.subplots_adjust(left=.09, right=.98, top=.87, bottom=.20, hspace=.24, wspace=.08)
    titles = ['Known elevation guide', 'Noisy TRAINING residual', 'Expected TRAINING target', 'Fresh GENERATION output']
    if name == 'structure-40': titles[-1] = 'Generated + fixed constraints'
    for row in range(3):
        for col, key in enumerate(['guide_metres', 'noisy_residual', 'target_metres', 'output_metres']):
            panel(axes[row, col], data[key][row, 0], 'noise' if col == 1 else 'elevation', spacing, titles[col] if row == 0 else '')
        axes[row, 0].set_yticks([0], [LABELS[row]], fontsize=9, rotation=90, va='center')
    fig.suptitle(title, fontsize=16, y=.985)
    fig.text(.5, .925, f'Actual saved train records; each panel {data["target_metres"].shape[-1]*spacing/1000:,.2f} km square. '
             'Patch orientations are local, not north-up.', ha='center', fontsize=10)
    fig.text(.5, .135, note, ha='center', fontsize=9)
    bars(fig, noise=True)
    fig.savefig(out/f'{name}.png', dpi=170)
    plt.close(fig)


@torch.no_grad()
def regional(out, device, metadata):
    folder = ROOT/'outputs/planet_training/data/bridge-adapted'
    t = tensor_records(folder, 'target', BRIDGE_IDS, device)
    c = tensor_records(folder, 'cond', BRIDGE_IDS, device)
    ctx = tensor_records(folder, 'context', BRIDGE_IDS, device)
    centres = np.load(folder/'train_centre.npy', mmap_mode='r')[BRIDGE_IDS].tolist()
    metadata['bridge_examples'] = {'split': 'train', 'indices': BRIDGE_IDS, 'centres_lat_lon': centres}
    records = {}
    entries = [
        ('intermediate-selected', 'Selected intermediate: adapted 1,750 updates', 'outputs/intermediate/selected-adapted-1750/best.pt', RECIPE),
        ('intermediate-pilot', 'Unchanged-Earth intermediate pilot: 500 updates (not selected)', 'outputs/intermediate/structure-pilot-v1/latest.pt', STRUCTURE_RECIPE),
        ('meso-selected', 'Selected meso: adapted 1,000 updates', 'outputs/meso/selected-adapted-1000/best.pt', MESO_RECIPE),
    ]
    for name, title, checkpoint, recipe in entries:
        print(f'Sampling {name}: three small training examples', flush=True)
        model, norm, ck = load_checkpoint(ROOT/checkpoint, device)
        if name.startswith('meso'):
            mf = ROOT/'outputs/planet_training/data/meso-adapted'
            real, coarse, context = [tensor_records(mf, k, MESO_IDS, device) for k in ['target', 'cond', 'context']]
            meta = json.loads((mf/'manifest.json').read_text())
            with np.load(ROOT/'outputs/meso/dataset.npz') as source:
                source_ids = [meta['parent_source_indices']['train'][i] for i in MESO_IDS]
                mc = source['train_centre'][source_ids].tolist()
            metadata['meso_examples'] = {'split': 'train', 'indices': MESO_IDS, 'source_indices': source_ids, 'centres_lat_lon': mc}
            spacing = 2880.
        else:
            real, coarse, context = t, c, ctx
            spacing = 23040.
        target, guide = paired_degradation(norm, real, coarse, context,
            torch.Generator(device=device).manual_seed(2026), recipe=recipe, sigma_cells=1., clean_fraction=0.)
        # This jitter follows target adaptation, in the same order as training.
        guide = guide + torch.randn(guide.shape, device=device, generator=torch.Generator(device=device).manual_seed(2027))*norm.s['cond_std']*ck.get('cond_noise', .03)
        up, cin = norm.condition(guide)
        x = norm.residual(target, up)
        noisy = x + torch.randn(x.shape, device=device, generator=torch.Generator(device=device).manual_seed(2028))*.5
        # Generation receives ONLY the guide and fresh noise, not `noisy` or `target`.
        samples = [sample_patches(model, norm, guide[i:i+1], steps=32, seed=42, device=device)[0] for i in range(3)]
        data = dict(original_metres=array(decode(real)), guide_metres=array(decode(up)),
                    target_metres=array(decode(target)), output_metres=array(decode(torch.cat(samples))),
                    noisy_residual=array(noisy), clean_residual=array(x), condition_model_units=array(cin), coarse_encoded=array(guide))
        records[name] = data
        np.savez_compressed(out/f'{name}.npz', **data)
        metadata['models'][name] = dict(checkpoint=str(ROOT/checkpoint), sha256=digest(ROOT/checkpoint),
            step=ck.get('step'), spacing_m=spacing, guide_blur_sigma_coarse_cells=1., condition_jitter=ck.get('cond_noise', .03), recipe=recipe)
        note = 'Noisy example: sigma = 0.5. Generated samples: 32 steps, seed 42 separately per row; target is never supplied to generation.'
        regional_figure(out, name, title, data, spacing, note)
        del model, ck

    print('Sampling structure-40: three small training examples', flush=True)
    from planet_training.terrain_structure import StructureData, target_for, load, sample
    checkpoint = ROOT/'outputs/structure/terrain-40-pilot-v1/review-20261013-182311-823544/structure-checkpoint.pt'
    model, norm, ck = load(checkpoint, device)
    ds = StructureData(ROOT/'outputs/planet_training/data/terrain-structure-40-v1')
    raw, coarse, context = ds.batch(3, indices=BRIDGE_IDS, device=device)
    coarse = degrade_guides(context, coarse, torch.Generator(device=device).manual_seed(2026), sigma_cells=1., clean_fraction=0., recipe=STRUCTURE_RECIPE)
    target, base = target_for(raw, coarse)
    _, up, cin = norm.condition(coarse)
    x = norm.residual(target, up)
    noisy = x + torch.randn(x.shape, device=device, generator=torch.Generator(device=device).manual_seed(2028))*.5
    samples = [sample(model, norm, coarse[i:i+1], steps=32, seed=42, device=device) for i in range(3)]
    data = dict(original_metres=array(raw), guide_metres=array(base), target_metres=array(decode(target)),
                output_metres=array(torch.cat([s[0] for s in samples])), raw_output_metres=array(torch.cat([s[2] for s in samples])),
                eligibility=array(eligibility(base)), noisy_residual=array(noisy), clean_residual=array(x), condition_model_units=array(cin), coarse_encoded=array(coarse))
    records['structure-40'] = data
    np.savez_compressed(out/'structure-40.npz', **data)
    metadata['models']['structure-40'] = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), step=ck.get('step'), spacing_m=40320.,
        guide_blur_sigma_coarse_cells=1., condition_jitter=0., recipe=STRUCTURE_RECIPE, output='raw learned output plus fixed regional projection; both saved')
    regional_figure(out, 'structure-40', 'New 40 km structure pilot: 2,000 updates (not selected)', data, 40320.,
        'A second known input marks where terrain may change (shown separately). Final output includes fixed local constraints; sea level remains zero.')

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.20, hspace=.24, wspace=.08)
    for row in range(3):
        for col, key in enumerate(['guide_metres', 'eligibility', 'raw_output_metres', 'output_metres']):
            panel(axes[row, col], data[key][row, 0], 'eligibility' if col == 1 else 'elevation', 40320.,
                ['Known elevation guide', 'Known permission mask', 'Raw learned model output', 'After fixed constraints'][col] if row == 0 else '')
        axes[row, 0].set_yticks([0], [LABELS[row]], fontsize=9, rotation=90, va='center')
    fig.suptitle('40 km pilot: model output versus deterministic constraints', fontsize=16, y=.98)
    fig.text(.5, .93, 'Permission mask: black = locked at input height; white = free; grey = feathered. Each panel 2,580.48 km square.', ha='center', fontsize=10)
    fig.text(.5, .135, 'Constraints protect input elevations at or below 200 m and conserve local 645.12 km means. This is not a sea-level shift.', ha='center', fontsize=9)
    bars(fig)
    fig.savefig(out/'structure-constraints.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.20, hspace=.24, wspace=.08)
    for row in range(3):
        values = [records['intermediate-selected']['original_metres'][row, 0],
                  records['intermediate-selected']['target_metres'][row, 0],
                  records['intermediate-pilot']['target_metres'][row, 0], records['structure-40']['target_metres'][row, 0]]
        for col, z in enumerate(values):
            panel(axes[row, col], z, 'elevation', 40320. if col == 3 else 23040.,
                ['Original modern Earth', 'Selected intermediate target', '500-update pilot target', '40 km pilot target'][col] if row == 0 else '')
        axes[row, 0].set_yticks([0], [LABELS[row]], fontsize=9, rotation=90, va='center')
    fig.suptitle('What the training labels actually ask the models to reproduce', fontsize=16, y=.98)
    fig.text(.5, .93, 'Same source records and 2,580.48 km footprint. All guides use the same one-coarse-cell smoothing.', ha='center', fontsize=10)
    fig.text(.5, .135, 'Selected target: old Earth residual on a smoothed baseline. 500-update pilot: unchanged Earth. 40 km: constrained Earth.', ha='center', fontsize=9)
    bars(fig); fig.savefig(out/'target-comparison.png', dpi=170); plt.close(fig)
    return records


@torch.no_grad()
def global_examples(out, device, metadata):
    from planet_training.structured_data import StructuredData, decode_relief
    from planet_training.data import PaleoRaster, layout_target
    from planet_training.generate_structured import load
    from planet_training.train_structured import sample
    from planet_training.geography import export_pixels
    selection = json.loads((ROOT/'outputs/continents/structured-selection.json').read_text())
    ds = StructuredData(ROOT/'outputs/planet_training/data/layout')
    i = next(i for i, r in enumerate(ds.meta['records']) if r['age_ma'] == 60 and r['split'] == 'train')
    z = layout_target(PaleoRaster(ds.source.maps[i]), np.eye(3), ds.n, ds.meta['supersample'])
    values = torch.from_numpy(z)[None].to(device)
    metadata['paleo_example'] = dict(record=ds.meta['records'][i], index=i, rotation='identity, allowed by random-rotation augmentation')
    results = {}
    for stage in ['coast', 'relief']:
        print(f'Sampling {stage}: one coarse globe, not intermediate/meso planet generation', flush=True)
        path = selection[stage]['checkpoint']
        model, ck, sha = load(path, stage, device)
        if sha != selection[stage]['checkpoint_sha256']: raise ValueError('Selected checkpoint hash changed')
        coast, magnitude = ds.encode_batch(values, ck['stats'], device)
        coverage = model.grid.mean(coast)[:, 0]
        x = coast-.5 if stage == 'coast' else magnitude
        noisy = x + model.grid.noise(1, 1, device, torch.Generator(device=device).manual_seed(2028))*.5
        generated = sample(model, ck['stats'], coverage, 32, 42, coast if stage == 'relief' else None)
        clean = coast[0, 0] if stage == 'coast' else values[0].abs().square()
        output = (generated[0, 0] > .5).float() if stage == 'coast' else generated[0, 0].square()
        fields = dict(coast=array(coast[0, 0]), noisy=array(noisy[0, 0]), target=array(clean), output=array(output),
                      raw_sample=array(generated[0, 0]))
        if stage == 'relief':
            fields['signed_target'] = decode_relief(fields['coast'], np.abs(z))
            fields['signed_output'] = decode_relief(fields['coast'], fields['raw_sample'])
        np.savez_compressed(out/f'{stage}.npz', **fields)
        results[stage] = fields
        metadata['models'][stage] = dict(checkpoint=path, sha256=sha, step=ck['step'], coverage=float(coverage[0]),
            relief_condition='true 60 Ma training mask, not generated coast' if stage == 'relief' else None)
        cols = 4 if stage == 'coast' else 5
        fig, axes = plt.subplots(1, cols, figsize=(cols*3.8, 4.7))
        fig.subplots_adjust(left=.025, right=.985, top=.72, bottom=.38, wspace=.08)
        titles = ['Known inputs', 'Noisy TRAINING field', 'Expected TRAINING target', 'Fresh GENERATION output']
        if stage == 'coast':
            axes[0].axis('off')
            axes[0].text(.5, .5, f'Land fraction: {float(coverage[0])*100:.2f}%\nNoise level: sigma = 0.5\n\nNo map is supplied to coast.', ha='center', va='center', fontsize=12)
        else:
            axes[0].imshow(export_pixels(fields['coast'], 256) > .5, cmap=MASK_CMAP, vmin=0, vmax=1)
            titles = ['Known coast + land fraction', 'Noisy TRAINING magnitude', 'Expected |elevation|', 'Generated |elevation|', 'Coast supplies output sign']
        axes[1].imshow(export_pixels(fields['noisy'], 256), cmap='RdBu_r', norm=NOISE_NORM)
        for col, key in [(2, 'target'), (3, 'output')]:
            zz = export_pixels(fields[key], 256)
            if stage == 'coast': axes[col].imshow(zz > .5, cmap=MASK_CMAP, vmin=0, vmax=1)
            else: axes[col].imshow(zz, cmap=CMAP, norm=NORM)
        if stage == 'relief': axes[4].imshow(export_pixels(fields['signed_output'], 256), cmap=CMAP, norm=NORM)
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{stage.capitalize()} model: 10,000 updates; real 60 Ma training example', fontsize=16, y=.97)
        fig.text(.5, .83, 'Whole world, equirectangular display; native spherical spacing about 161 km. No global hillshading.', ha='center', fontsize=10)
        fig.text(.5, .25, 'Generation starts from fresh noise: 32 steps, seed 42. It is a new sample, not a reconstruction of the noisy training example.', ha='center', fontsize=10)
        bars(fig, noise=True, coast=stage == 'coast', global_view=True)
        fig.savefig(out/f'{stage}.png', dpi=170); plt.close(fig)
        del model, ck
    return results


def overview(out, regional_records):
    names = ['intermediate-selected', 'intermediate-pilot', 'structure-40', 'meso-selected']
    labels = ['Selected intermediate\n1,750 adapted updates\n23.04 km cells', 'Intermediate pilot\n500 updates; not selected\n23.04 km cells',
              'Structure pilot\n2,000 updates; not selected\n40.32 km cells', 'Selected meso\n1,000 adapted updates\n2.88 km cells']
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    fig.subplots_adjust(left=.22, right=.97, top=.91, bottom=.16, hspace=.26, wspace=.07)
    for row, name in enumerate(names):
        spacing = 40320. if name == 'structure-40' else 2880. if name == 'meso-selected' else 23040.
        d = regional_records[name]
        for col, key in enumerate(['guide_metres', 'target_metres', 'output_metres']):
            panel(axes[row, col], d[key][0, 0], 'elevation', spacing,
                ['Known guide', 'Expected training target', 'Fresh output'][col] if row == 0 else '')
        axes[row, 0].set_yticks([0], [labels[row]], fontsize=10, rotation=0, va='center')
    fig.suptitle('Regional models: given terrain, training target, generated terrain', fontsize=16, y=.98)
    fig.text(.5, .94, 'Mountainous training records. First three rows share a 2,580.48 km footprint; meso shows its 230.40 km training crop.', ha='center', fontsize=10)
    fig.text(.5, .11, '40 km output includes fixed local constraints. Noisy training input and second condition channel appear in the detailed pages.', ha='center', fontsize=9)
    bars(fig); fig.savefig(out/'regional-overview.png', dpi=170); plt.close(fig)


def report(out, metadata):
    text = """# Training inputs, targets and actual model outputs

These are examples from the **saved training split**, not held-out quality scores. Outputs were freshly sampled using the checkpoint listed in `provenance.json`; no training or full intermediate/meso globe was run. All examples use 32 sampling steps and seed 42. Regional examples restart that seed separately so differences are due to their conditions/models. They are illustrative samples, not averages over seeds.

## How to read these comparisons

The elevation guide is shown in metres for readability. The models receive encoded/normalized heights, not RGB terrain images. During training a model also receives a **noisy clean target** (coast/relief) or **noisy target-minus-guide residual** (regional models), and its noise level. It learns to recover the clean field. During generation the clean target is unavailable: sampling starts from fresh noise. Thus the displayed noisy training input is **not** used to produce the final column.

The square-root encoding is sign(z) × sqrt(abs(z)); relief uses unsigned sqrt(abs(z)). Noise panels show model units, not metres. A single illustrative training noise level, sigma = 0.5, is used; actual training draws many noise levels. Input preprocessing inside EDM also rescales by sigma. Land masks are displayed as discrete masks after sampling onto the world display grid.

## Coast

**Given:** noisy coast scores, noise level and one land-fraction number. There is no continent map supplied as a guide. **Expected:** the PaleoDEM land/ocean mask at about 161 km spherical spacing. **Created:** new continent geometry consistent with the scalar condition, not a copy of the target planet. The displayed 60 Ma target is from a training age; its original orientation is one valid augmentation orientation.

![Coast](coast.png)

## Unsigned relief

**Given:** known coastline, land fraction, noise level and noisy unsigned relief. **Expected:** PaleoDEM sqrt(abs(elevation)) at the same coarse scale. The target and generated magnitude panels decode this unsigned field into metres. **Created:** a fresh magnitude field. The fixed coast supplies its positive land/negative ocean sign in the last panel. This uses the **true training coast** to isolate relief; end-to-end production uses a sampled coast. No datum shift is applied.

![Relief](relief.png)

## Regional overview

The first three rows use the exact same saved mountainous source record. The meso row is a different real training patch at its much smaller physical footprint.

![Regional overview](regional-overview.png)

## Selected intermediate: 161.28 km guide → 23.04 km cells

**Given:** a coarse modern-Earth guide with training-time smoothing/jitter, bilinearly expanded in encoded elevation, plus a noisy residual and noise level. **Expected:** the original Earth residual placed on the degraded baseline, **not unchanged Earth heights**. **Created:** a fresh residual added to that guide. Training patches have 112 × 112 target cells (2,580.48 km square); guide has 18 × 18 cells including margins.

![Selected intermediate](intermediate-selected.png)

## Unchanged-target intermediate pilot: 500 updates, not selected

**Given:** the same kind of degraded coarse guide, noisy residual and noise level. **Expected:** unchanged modern Earth at 23.04 km, so the residual must restore broad belt/valley structure removed from the guide. **Created:** the sampled residual from the pilot checkpoint. These paired examples make the changed objective visible; the pilot also differs in training history and is not a controlled demonstration that target changes alone caused the result.

![Intermediate pilot](intermediate-pilot.png)

## New structure pilot: 161.28 km guide → 40.32 km cells

**Given:** the guide interpolated in metres, an additional permission mask derived from its height, a noisy residual and noise level. **Expected:** modern 40.32 km Earth averages modified to protect input heights ≤200 m and conserve regional 645.12 km means. The 200–600 m band feathers changes. **Created:** a learned residual, followed by fixed local constraints. Target grids are 64 × 64 cells, covering the same 2,580.48 km footprint. This pilot remains unselected.

![Structure pilot](structure-40.png)

The next page separates learned output from the constraint operation. Low coastal heights and marine shelves are locked at input heights; that is not a global elevation offset. The mask is a model input. In encoded form its second channel is 2 × permission − 1.

![Structure constraints](structure-constraints.png)

## Selected meso: 23.04 km guide → 2.88 km cells

**Given:** the smoothed/jittered coarse guide, noisy residual and noise level. **Expected:** original modern-Earth fine residual over the degraded baseline, analogous to the selected intermediate objective. **Created:** a sampled residual added to the guide. These are the actual 80 × 80 training crops (230.40 km square), not the 128 × 128 production tiles. The guide has 12 × 12 cells including margins. These outputs are the raw learned model result; the experimental smooth conservation operation is not applied here.

![Selected meso](meso-selected.png)

## Original Earth versus actual labels

All columns show the same source geography. This is the key distinction between raw Earth references and the terrain the models are actually supervised to produce.

![Training labels](target-comparison.png)

## Reproducibility and limits

The regional figures illustrate the smoothed 75% branch of training: Gaussian sigma = 1 coarse cell, with real surrounding context. Training also uses 25% unmodified guides and randomly varies smoothing; the intermediate/meso conditioning jitter is copied from each checkpoint. The 40 km pilot does not add that jitter. Saved array values include finite half-precision rounding where present. Arrays in adjacent `.npz` files retain physical/encoded/model fields and raw versus constrained 40 km outputs. `provenance.json` records source indices, centres, scales and checkpoint hashes.

All elevation panels use the same −6,000 to +6,000 m colour limits and fixed NW illumination with 5× vertical shading; noise and binary masks have separate colours. World maps have no shading because local physical pixel size varies with latitude. Patch orientations are the stored random local frames, not north-up. These examples show the learning task and sampled behavior; they do not assert exact reproduction of Earth features or replace the multi-seed regional assessment.
"""
    (out/'README.md').write_text(text, encoding='utf-8')
    sections = [('coast','Coast'), ('relief','Relief'), ('regional-overview','Regional overview'),
                ('intermediate-selected','Selected intermediate'), ('intermediate-pilot','Intermediate pilot'),
                ('structure-40','40 km pilot'), ('structure-constraints','40 km constraints'),
                ('meso-selected','Selected meso'), ('target-comparison','Actual training labels')]
    links = ''.join(f'<a href="#{name}">{title}</a>' for name, title in sections)
    descriptions = {
        'coast': 'Given: land fraction and a noisy coast field. Target: PaleoDEM land/ocean classes. Output: new continent geometry. No map guide is given to this model.',
        'relief': 'Given: known coast, land fraction and noisy unsigned relief. Target: PaleoDEM elevation magnitude. Output: a fresh magnitude field; the coast fixes its land/ocean sign.',
        'regional-overview': 'First three rows share one source patch and physical footprint. The meso training crop is much smaller. Full examples below also show lowlands and coasts.',
        'intermediate-selected': 'Target: original modern-Earth detail added to the smoothed guide baseline. This target differs from unchanged Earth elevations. Output: a fresh sampled residual added to the guide.',
        'intermediate-pilot': 'Target: unchanged modern Earth, including broad corrections missing from the smoothed guide. This 500-update pilot is not selected.',
        'structure-40': 'Given: an elevation guide plus a permission mask. Target: constrained modern Earth at 40.32 km. Output shown here includes the fixed constraints. This pilot is not selected.',
        'structure-constraints': 'The permission mask is the second condition channel. Black cells are locked; white cells may change; grey cells feather changes. Raw learned output and constrained output are shown separately.',
        'meso-selected': 'Target: original fine Earth residual placed on a degraded guide. These are 80-cell training crops, 230.40 km across. Output is the raw model sample, without the experimental conservation operation.',
        'target-comparison': 'The expected training label is not always the original Earth raster. This page shows the selected intermediate translation, unchanged-Earth pilot and constrained 40 km label on the same source records.'}
    content = ''.join(f'<section id="{name}"><h2>{title}</h2><p>{descriptions[name]}</p><a href="{name}.png"><img src="{name}.png" alt="{title}"></a></section>' for name,title in sections)
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Terrain training examples</title>'
        '<style>body{font:16px system-ui;background:#f4f4f4;margin:0;color:#222}header{padding:18px 30px;background:white}nav{position:sticky;top:0;display:flex;gap:15px;flex-wrap:wrap;background:#fff;padding:12px 30px;border-bottom:1px solid #ccc}a{color:#265d81}section{padding:20px 30px;scroll-margin-top:90px}img{width:100%;max-width:1900px}p{max-width:1100px}h2{font-size:23px}</style></head><body>'
        '<header><h1>What each terrain model receives and learns</h1><p>Saved training examples beside fresh model samples. '
        'Read left to right: known guide, noisy training input, expected training target, fresh generation. '
        'Generation starts from fresh noise; it does not use the displayed training target. Click an image for full size.</p>'
        '<p><a href="README.md">Full explanations</a> · <a href="provenance.json">Model and sample provenance</a></p></header><nav>'+links+'</nav>'+content+'</body></html>', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='outputs/model-training-io')
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = pick_device(a.device)
    if device == 'cpu': raise ValueError('Fresh sampling intended as a cheap CUDA test; use the GPU environment')
    start = time.perf_counter()
    metadata = dict(device=device, sigma_training_example=.5, sampler_steps=32, sample_seed=42, models={})
    global_records = global_examples(out, device, metadata)
    records = regional(out, device, metadata)
    overview(out, records)
    for name, d in {**global_records, **records}.items():
        if not all(np.isfinite(v).all() for v in d.values()): raise ValueError(f'Nonfinite field: {name}')
    np.testing.assert_array_equal(records['intermediate-pilot']['target_metres'], records['intermediate-pilot']['original_metres'])
    from planet_training.terrain_structure import constraint_metrics
    s = records['structure-40']
    metadata['checks'] = {'all_saved_fields_finite': True, 'unchanged_pilot_target_exactly_matches_source': True}
    for key in ['target_metres', 'output_metres']:
        m = constraint_metrics(torch.from_numpy(s[key]), torch.from_numpy(s['guide_metres']), 16, 40320.)
        # Encoding then decoding the target can introduce sub-millimetre FP32
        # rounding at protected depths, even though projection locks them.
        if m['protected_height_change_max_m'] > .001 or m['sign_disagreement_fraction'] != 0 or m['added_footprint_mean_drift_max_m'] > .01:
            raise ValueError(f'Structure constraints failed: {key}: {m}')
        metadata['checks'][f'structure_{key}'] = m
    metadata['elapsed_seconds'] = time.perf_counter()-start
    (out/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report(out, metadata)
    print(f'Finished in {metadata["elapsed_seconds"]:.1f} seconds: {out}', flush=True)


if __name__ == '__main__': main()
