"""Paired target-oracle and genuinely chained regional terrain flows. No training."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_terrain_structure import image, CMAP, NORM
from scripts.explain_model_training_io import digest
from scripts.review_intermediate_structure import frame
from planet_training.data import bridge_patch, PaleoRaster, layout_target
from planet_training.adaptation import paired_degradation, coarse_context, RECIPE
from planet_training.terrain_structure import decode, encode, conservative_resample, target_for, handoff
from planet_training.smooth_conservation import project_smooth
from toy_meso.common import load_checkpoint, sample_patches
from toy_meso.sphere import GlobalRaster, area_average_patch, block_mean

CASES = [('alps', 'Alps', 45.5, 9.), ('california', 'California: main and secondary ranges', 37., -120.),
         ('plains', 'Great Plains / eastern Rocky Mountain transition', 42., -103.)]
MASK = ListedColormap(['#16465e', '#a6af70'])


def arr(z):
    return z.detach().float().cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)


class WindowRaster:
    """Same pixel-centre bilinear lookup as GlobalRaster, reading bounded windows."""
    def __init__(self, path):
        self.ds = rasterio.open(path)
        self.h, self.w = self.ds.height, self.ds.width
        if self.w != 2*self.h: raise ValueError('Expected global 2:1 DEM')

    def sample(self, lat, lon):
        x = ((lon+180)/360*self.w-.5) % self.w
        y = np.clip((90-lat)/180*self.h-.5, 0, self.h-1)
        x0, y0 = np.floor(x).astype(np.int64), np.minimum(np.floor(y).astype(np.int64), self.h-2)
        if x0.min() < 0 or x0.max() >= self.w-1 or x0.max()-x0.min() > self.w//2:
            raise ValueError('This bounded reviewer requires a patch away from the dateline')
        left, top = int(x0.min()), int(y0.min())
        z = self.ds.read(1, window=Window(left, top, int(x0.max())-left+2, int(y0.max())-top+2)).astype(np.float32)
        if self.ds.nodata is not None and (z == self.ds.nodata).any(): raise ValueError('Nodata in reference patch')
        xx, yy = x0-left, y0-top
        fx, fy = x-x0, y-y0
        upper = z[yy, xx]*(1-fx)+z[yy, xx+1]*fx
        lower = z[yy+1, xx]*(1-fx)+z[yy+1, xx+1]*fx
        return (upper*(1-fy)+lower*fy).astype(np.float32)


def flow_figure(out, name, title, columns, ideal, actual, global_view=False, note='', input_note='', complete=False):
    count = len(columns)
    fig, axes = plt.subplots(2, count, figsize=(count*3.25+1.2, 7.6 if global_view else 8.3), squeeze=False)
    fig.subplots_adjust(left=.095, right=.985, top=.80, bottom=.23, hspace=.65, wspace=.17)
    for row, fields in enumerate([ideal, actual]):
        for col, z in enumerate(fields):
            ax = axes[row, col]
            if z is None:
                ax.axis('off'); ax.text(.5, .5, input_note, ha='center', va='center', fontsize=12)
            elif global_view and not (complete and col >= 3):
                ax.imshow(z, cmap=MASK if col == 1 else CMAP, vmin=0 if col == 1 else None,
                          vmax=1 if col == 1 else None, norm=None if col == 1 else NORM, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.imshow(image(z, 368640./len(z)), extent=(-184.32, 184.32, -184.32, 184.32), interpolation='nearest')
                ax.set_xticks([-184.32, 0, 184.32], ['−184', '0', '184'], fontsize=8)
                ax.set_yticks([]); ax.set_xlabel('km', fontsize=8)
            ax.set_title(columns[col], fontsize=11, pad=10)
        box = axes[row, 0].get_position()
        fig.text(.022, (box.y0+box.y1)/2, ['IDEAL\n(targets)', 'ACTUAL\n(samples)'][row],
                 fontsize=11, weight='bold', ha='left', va='center', color=['#32674a', '#905a2d'][row])
        for col in range(count-1):
            a, b = axes[row, col].get_position(), axes[row, col+1].get_position()
            y = (a.y0+a.y1)/2
            fig.add_artist(FancyArrowPatch((a.x1+.004, y), (b.x0-.004, y), transform=fig.transFigure,
                arrowstyle='-|>', mutation_scale=14, linewidth=1.6,
                linestyle='--' if complete and row == 0 and col == 2 else '-',
                color='#ad6c28' if complete and row == 0 and col == 2 else '#65717b'))
            if complete and row == 0 and col == 2:
                fig.text((a.x1+b.x0)/2, y+.045, 'training\nsource changes', fontsize=8, ha='center', color='#ad6c28')
    fig.suptitle(title, fontsize=17, y=.975)
    fig.text(.5, .895, 'First maps: whole world. Last three maps: one 368.64 km regional zoom, with full model context.' if complete else
             'Whole-world prefix: coast then relief. Finer stages continue in the regional flows.' if global_view else
             'Every terrain panel shows the SAME central 368.64 km square. Models receive their full surrounding context.',
             ha='center', fontsize=11)
    fig.text(.5, .15, note, ha='center', fontsize=10)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=fig.add_axes([.21, .087, .60, .014]), orientation='horizontal')
    cb.set_ticks([-6000, -3000, 0, 1000, 2000, 4000, 6000]); cb.ax.tick_params(labelsize=8)
    cb.set_label('Elevation / magnitude (m); fixed colours. '+('World maps unshaded; regional maps NW light, 5× shading.' if complete else
        'World maps unshaded.' if global_view else 'NW light; 5× shading.'), fontsize=9)
    fig.savefig(out/f'{name}.png', dpi=160); plt.close(fig)


def zoom23(z):
    return arr(z)[0, 0, 48:64, 48:64]


def zoom40(z):
    return zoom23(F.interpolate(z.float(), size=(112, 112), mode='bilinear', align_corners=False))


@torch.no_grad()
def regions(out, metadata):
    device = 'cuda'
    ids = metadata['models']
    intermediate, ni, _ = load_checkpoint(ids['intermediate-selected']['checkpoint'], device)
    meso, nm, _ = load_checkpoint(ids['meso-selected']['checkpoint'], device)
    from planet_training.terrain_structure import load, sample, constraint_metrics
    structure, ns, _ = load(ids['structure-40']['checkpoint'], device)
    earth = GlobalRaster(np.load(ROOT/'outputs/planet_training/data/cache/etopo-filtered.npy', mmap_mode='r'))
    fine_source = WindowRaster(ROOT/'data/etopo_2022_30s_surface.tif')
    cards = []
    for name, title, lat, lon in CASES:
        print(f'Round trip: {name}', flush=True)
        f = frame(lat, lon)
        source23, original_c = bridge_patch(earth, f)
        t = torch.from_numpy(source23.astype(np.float32))[None, None].to(device)
        c = torch.from_numpy(original_c)[None, None].to(device)
        ctx = torch.from_numpy(coarse_context(earth, f, 112))[None, None].to(device)
        ideal23, guide = paired_degradation(ni, t, c, ctx, None, recipe=RECIPE, sigma_cells=1., clean_fraction=0.)
        sampled23, up23 = sample_patches(intermediate, ni, guide, 32, 42, device)
        sampled23 = sampled23.float()

        # Meso's original fine target and original coarse guide, with real halo.
        # 192 = 128 target + 2*(one model margin + 3 recipe halo)*8.
        raw192, _, _ = area_average_patch(fine_source, f, 192, 2880., 3)
        fine_target = torch.from_numpy(np.sign(raw192[32:-32, 32:-32])*np.sqrt(np.abs(raw192[32:-32, 32:-32])))
        fine_target = fine_target.half().float()[None, None].to(device)
        fine_c = block_mean(raw192, 8)[3:-3, 3:-3]
        fine_c = torch.from_numpy(np.sign(fine_c)*np.sqrt(np.abs(fine_c)))[None, None].to(device)
        original_fine_up, _ = nm.condition(fine_c)

        def expected_meso(encoded23):
            incoming = encoded23[..., 47:65, 47:65].float()
            up, _ = nm.condition(incoming)
            # The SAME residual-preservation rule as the selected meso recipe.
            target = fine_target+up-original_fine_up
            return decode(target), decode(up), incoming

        target_fine, _, ideal_meso_c = expected_meso(ideal23)
        actual_meso_c = sampled23[..., 47:65, 47:65].float()
        fine_sample, _ = sample_patches(meso, nm, actual_meso_c, 32, 42, device)
        sampled_fine = decode(fine_sample.float())
        source_fine = raw192[32:-32, 32:-32]
        data = dict(reference_fine_metres=source_fine, reference23_metres=arr(decode(t)), shared_coarse_encoded=arr(guide),
            shared_guide23_metres=arr(decode(up23)), ideal_intermediate_encoded=arr(ideal23),
            actual_intermediate_encoded=arr(sampled23), ideal_meso_input_encoded=arr(ideal_meso_c),
            actual_meso_input_encoded=arr(actual_meso_c), ideal_meso_metres=arr(target_fine), actual_meso_metres=arr(sampled_fine))
        np.testing.assert_array_equal(data['actual_meso_input_encoded'], data['actual_intermediate_encoded'][..., 47:65, 47:65])
        flow_figure(out, name+'-selected', title+' — selected pipeline round trip',
            ['Modern Earth source\nreference', 'Shared relief guide\n161 km cells', 'Intermediate\n23 km cells', 'Meso\n2.9 km cells'],
            [source_fine, zoom23(decode(up23)), zoom23(decode(ideal23)), arr(target_fine)[0, 0]],
            [source_fine, zoom23(decode(up23)), zoom23(decode(sampled23)), arr(sampled_fine)[0, 0]],
            note='Only the coarse guide reaches generation. Actual meso receives actual intermediate output; no Earth reset. Seed 42, 32 steps.')
        cards.append((name+'-selected', title+' · selected pipeline'))

        real40 = conservative_resample(decode(t)[:, 0], 64)[:, None]
        ideal40, base40 = target_for(real40, guide)
        actual40, _, raw40 = sample(structure, ns, guide, 32, 42, device)
        ideal_handoff23, _, base23 = handoff(decode(ideal40), base40, ideal23, guide)
        actual_handoff23, _, _ = handoff(actual40, base40, sampled23, guide)
        structure_target_fine, ideal_fine_base, ideal_structure_meso_c = expected_meso(encode(ideal_handoff23))
        ideal_final = project_smooth(structure_target_fine, ideal_fine_base, 8, 2880.)
        actual_structure_meso_c = encode(actual_handoff23[..., 47:65, 47:65]).float()
        actual_fine, actual_fine_up = sample_patches(meso, nm, actual_structure_meso_c, 32, 42, device)
        actual_final = project_smooth(decode(actual_fine.float()), decode(actual_fine_up), 8, 2880.)
        data.update(ideal40_metres=arr(decode(ideal40)), actual40_metres=arr(actual40), raw40_metres=arr(raw40),
            ideal_handoff23_metres=arr(ideal_handoff23), actual_handoff23_metres=arr(actual_handoff23),
            ideal_structure_meso_input_encoded=arr(ideal_structure_meso_c), actual_structure_meso_input_encoded=arr(actual_structure_meso_c),
            ideal_structure_meso_training_target_metres=arr(structure_target_fine), ideal_structure_final_metres=arr(ideal_final),
            actual_structure_meso_raw_metres=arr(decode(actual_fine.float())), actual_structure_final_metres=arr(actual_final))
        np.testing.assert_array_equal(data['actual_structure_meso_input_encoded'], arr(encode(actual_handoff23[..., 47:65, 47:65]).float()))
        flow_figure(out, name+'-structure', title+' — experimental structure round trip',
            ['Modern Earth source\nreference', 'Shared relief guide\n161 km cells', 'Structure stage\n40 km cells', 'Intermediate handoff\n23 km cells', 'Meso + constraints\n2.9 km cells'],
            [source_fine, zoom23(decode(up23)), zoom40(decode(ideal40)), zoom23(ideal_handoff23), arr(ideal_final)[0, 0]],
            [source_fine, zoom23(decode(up23)), zoom40(actual40), zoom23(actual_handoff23), arr(actual_final)[0, 0]],
            note='Ideal handoff uses target intermediate detail; actual handoff uses sampled intermediate detail. Same fixed constraints in both rows.')
        cards.append((name+'-structure', title+' · 40 km pilot'))
        if not all(np.isfinite(v).all() for v in data.values()): raise ValueError('Nonfinite round trip')
        np.savez_compressed(out/f'{name}.npz', **data)
        metadata['regions'].append(dict(name=name, latitude=lat, longitude=lon,
            shared_input_identical=True, actual_meso_exactly_receives_actual_intermediate=True,
            actual_structure_meso_exactly_receives_actual_handoff=True,
            structure_constraint=constraint_metrics(actual40, base40, 16, 40320.)))
    fine_source.ds.close()
    return cards


@torch.no_grad()
def globals_(out, metadata):
    from planet_training.structured_data import StructuredData, decode_relief
    from planet_training.generate_structured import load
    from planet_training.train_structured import sample
    from planet_training.geography import export_pixels
    ds = StructuredData(ROOT/'outputs/planet_training/data/layout')
    coast_model, cc, _ = load(metadata['models']['coast']['checkpoint'], 'coast', 'cuda')
    relief_model, rc, _ = load(metadata['models']['relief']['checkpoint'], 'relief', 'cuda')
    cards = []
    for age in [60, 200]:
        print(f'Global prefix: PaleoDEM {age} Ma', flush=True)
        idx = next(i for i, r in enumerate(ds.meta['records']) if r['age_ma'] == age and r['split'] == 'train')
        z = layout_target(PaleoRaster(ds.source.maps[idx]), np.eye(3), ds.n, ds.meta['supersample'])
        encoded = torch.from_numpy(z)[None].to('cuda')
        coast, _ = ds.encode_batch(encoded, cc['stats'], 'cuda')
        coverage = coast_model.grid.mean(coast)[:, 0]
        scores = sample(coast_model, cc['stats'], coverage, 32, 42)
        sampled_coast = (scores > .5).float()
        sampled_magnitude = sample(relief_model, rc['stats'], coverage, 32, 42, sampled_coast)
        c, m = arr(sampled_coast[0, 0]), arr(sampled_magnitude[0, 0])
        actual_signed = decode_relief(c, m)
        ideal_signed = np.sign(z)*z*z
        fields = dict(ideal_coast=arr(coast[0, 0]), ideal_magnitude_metres=np.abs(ideal_signed), ideal_signed_metres=ideal_signed,
            actual_coast=c, actual_magnitude_metres=m*m, actual_signed_metres=actual_signed, actual_relief_condition=c)
        np.testing.assert_array_equal(fields['actual_coast'], fields['actual_relief_condition'])
        np.savez_compressed(out/f'paleo-{age}.npz', **fields)
        flow_figure(out, f'paleo-{age}', f'Global start: {age} Ma training target versus actual coast → relief',
            ['Starting condition', 'Coast\nland / ocean', 'Relief\nunsigned height / depth', 'Coast + relief\nsigned elevation'],
            [None, export_pixels(fields['ideal_coast'], 256) > .5, export_pixels(fields['ideal_magnitude_metres'], 256), export_pixels(ideal_signed, 256)],
            [None, export_pixels(c, 256) > .5, export_pixels(m*m, 256), export_pixels(actual_signed, 256)], global_view=True,
            input_note=f'Land fraction\n{float(coverage[0])*100:.2f}%\n\nGeneration starts\nfrom noise.',
            note='Actual relief receives the GENERATED coast, not the PaleoDEM mask. Sign comes from that coast. Sea level stays at zero.')
        metadata['global_prefixes'].append(dict(age_ma=age, training_record=idx, coverage=float(coverage[0]),
            relief_receives_generated_coast=True, original_record=ds.meta['records'][idx]))
        cards.append((f'paleo-{age}', f'Global start · {age} Ma'))
    return cards


def redraw_saved(out, metadata):
    """Improve presentation without repeating samples already saved."""
    cards = []
    for record in metadata['global_prefixes']:
        from planet_training.geography import export_pixels
        age = record['age_ma']
        with np.load(out/f'paleo-{age}.npz') as z:
            flow_figure(out, f'paleo-{age}', f'Global start: {age} Ma training target versus actual coast → relief',
                ['Starting condition', 'Coast\nland / ocean', 'Relief\nunsigned height / depth', 'Coast + relief\nsigned elevation'],
                [None, export_pixels(z['ideal_coast'], 256) > .5, export_pixels(z['ideal_magnitude_metres'], 256), export_pixels(z['ideal_signed_metres'], 256)],
                [None, export_pixels(z['actual_coast'], 256) > .5, export_pixels(z['actual_magnitude_metres'], 256), export_pixels(z['actual_signed_metres'], 256)],
                global_view=True, input_note=f'Land fraction\n{record["coverage"]*100:.2f}%\n\nGeneration starts\nfrom noise.',
                note='Actual relief receives the GENERATED coast, not the PaleoDEM mask. Sign comes from that coast. Sea level stays at zero.')
        cards.append((f'paleo-{age}', f'Global start · {age} Ma'))
    for name, title, _, _ in CASES:
        with np.load(out/f'{name}.npz') as d:
            source = d['reference_fine_metres']
            guide = zoom23(d['shared_guide23_metres'])
            dec = lambda key: decode(torch.from_numpy(d[key]))
            flow_figure(out, name+'-selected', title+' — selected pipeline round trip',
                ['Modern Earth source\nreference', 'Shared relief guide\n161 km cells', 'Intermediate\n23 km cells', 'Meso\n2.9 km cells'],
                [source, guide, zoom23(dec('ideal_intermediate_encoded')), d['ideal_meso_metres'][0, 0]],
                [source, guide, zoom23(dec('actual_intermediate_encoded')), d['actual_meso_metres'][0, 0]],
                note='Only the coarse guide reaches generation. Actual meso receives actual intermediate output; no Earth reset. Seed 42, 32 steps.')
            cards.append((name+'-selected', title+' · selected pipeline'))
            flow_figure(out, name+'-structure', title+' — experimental structure round trip',
                ['Modern Earth source\nreference', 'Shared relief guide\n161 km cells', 'Structure stage\n40 km cells', 'Intermediate handoff\n23 km cells', 'Meso + constraints\n2.9 km cells'],
                [source, guide, zoom40(torch.from_numpy(d['ideal40_metres'])), zoom23(d['ideal_handoff23_metres']), d['ideal_structure_final_metres'][0, 0]],
                [source, guide, zoom40(torch.from_numpy(d['actual40_metres'])), zoom23(d['actual_handoff23_metres']), d['actual_structure_final_metres'][0, 0]],
                note='Ideal handoff uses target intermediate detail; actual handoff uses sampled intermediate detail. Same fixed constraints in both rows.')
            cards.append((name+'-structure', title+' · 40 km pilot'))
    return cards


@torch.no_grad()
def complete_flows(out, metadata):
    """Actual full chain, contrasted with the two source families of supervision."""
    from planet_training.geography import export_pixels
    from planet_diffusion import cube
    model, norm, _ = load_checkpoint(metadata['models']['intermediate-selected']['checkpoint'], 'cuda')
    meso, mn, _ = load_checkpoint(metadata['models']['meso-selected']['checkpoint'], 'cuda')
    cards = []
    metadata.setdefault('complete_chains', [])
    for age, earth_name, label, target_height in [(60, 'alps', 'Generated highland', 2200.), (200, 'california', 'Generated upland', 900.)]:
        path = out/f'complete-{age}.npz'
        with np.load(out/f'paleo-{age}.npz') as p, np.load(out/f'{earth_name}.npz') as e:
            if path.exists():
                with np.load(path) as loaded: data = {key: loaded[key] for key in loaded.files}
            else:
                print(f'Complete chain: coast/relief {age} Ma condition -> {label.lower()} -> intermediate -> meso', flush=True)
                native = p['actual_signed_metres']
                display = export_pixels(native, 256)
                latitudes = 90-(np.arange(256)+.5)*180/256
                longitudes = -180+(np.arange(512)+.5)*360/512
                yy, xx = np.meshgrid(latitudes, longitudes, indexing='ij')
                distance = np.abs(display-target_height)
                distance[(abs(yy) > 45) | (abs(xx) > 150) | (display < 300)] = np.inf
                r, col = np.unravel_index(np.argmin(distance), distance.shape)
                lat, lon = float(yy[r, col]), float(xx[r, col])
                class NativeRaster:
                    def sample(self, lat, lon):
                        la, lo = np.radians(lat), np.radians(lon)
                        points = np.stack((np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)), axis=-1)
                        return cube.sample(native[None], points)[0]
                _, coarse = bridge_patch(NativeRaster(), frame(lat, lon))
                c = torch.from_numpy(coarse)[None, None].to('cuda')
                z, up = sample_patches(model, norm, c, 32, 42, 'cuda')
                c_meso = z[..., 47:65, 47:65].float()
                fine, _ = sample_patches(meso, mn, c_meso, 32, 42, 'cuda')
                data = dict(latitude=np.array(lat), longitude=np.array(lon), actual_intermediate_coarse_encoded=arr(c),
                    actual_guide23_metres=arr(decode(up)), actual_intermediate_encoded=arr(z.float()),
                    actual_meso_input_encoded=arr(c_meso), actual_meso_metres=arr(decode(fine.float())))
                np.testing.assert_array_equal(data['actual_meso_input_encoded'], data['actual_intermediate_encoded'][..., 47:65, 47:65])
                if not all(np.isfinite(v).all() for v in data.values()): raise ValueError('Nonfinite complete chain')
                np.savez_compressed(path, **data)
                metadata['complete_chains'].append(dict(age_condition_ma=age, latitude=lat, longitude=lon,
                    centre_selected_before_sampling='nearest positive coarse height to '+str(target_height)+' m, |lat|≤45, |lon|≤150',
                    intermediate_input='area means sampled directly from generated signed relief cube',
                    meso_input_exactly_generated_intermediate_crop=True, ideal_regional_training_source=earth_name,
                    ideal_training_source_changes=True, complete_actual_chain=True))
            coverage = next(r['coverage'] for r in metadata['global_prefixes'] if r['age_ma'] == age)
            title = f'Complete pipeline — {label.lower()} ({float(data["latitude"]):.1f}°, {float(data["longitude"]):.1f}°)'
            flow_figure(out, f'complete-{age}', title,
                ['Starting condition', 'Coast model\nwhole world', 'Relief model\nsigned whole world', 'Input to intermediate\nregional relief guide', 'Intermediate model\n23 km cells', 'Meso model\n2.9 km cells'],
                [None, export_pixels(p['ideal_coast'], 256) > .5, export_pixels(p['ideal_signed_metres'], 256),
                    zoom23(e['shared_guide23_metres']), zoom23(decode(torch.from_numpy(e['ideal_intermediate_encoded']))), e['ideal_meso_metres'][0, 0]],
                [None, export_pixels(p['actual_coast'], 256) > .5, export_pixels(p['actual_signed_metres'], 256),
                    zoom23(data['actual_guide23_metres']), zoom23(decode(torch.from_numpy(data['actual_intermediate_encoded']))), data['actual_meso_metres'][0, 0]],
                global_view=True, complete=True, input_note=f'Land fraction\n{coverage*100:.2f}%\n\nFresh noise\nat each stage.',
                note=f'Ideal: PaleoDEM labels → modern {earth_name.title()} target examples (source change marked). Actual: one generated world throughout; regional zoom, no resets.')
        cards.append((f'complete-{age}', label+' · complete pipeline'))
    return cards


def gallery(out, cards):
    descriptions = {
        'selected': 'Ideal: replace intermediate with its adapted training target, then apply the meso residual-preservation rule to that incoming terrain. Actual: generate intermediate, then give exactly that generated patch to meso. Both rows start with the same smoothed coarse guide.',
        'structure': 'Ideal: use constrained modern 40 km targets and the expected intermediate detail, then the meso target plus fixed final constraints. Actual: sampled 40 km structure and sampled intermediate detail feed the handoff, whose output feeds meso. This pilot is not selected.',
        'global': 'Ideal: actual PaleoDEM training labels. Actual: coast is sampled from noise and its generated mask conditions relief. These are different worlds; they are not expected to copy the training geography.',
        'complete': 'Ideal: the training target examples along the whole pipeline; the dashed arrow explicitly marks the switch from PaleoDEM supervision to modern Earth supervision. These are not geographically paired after that boundary. Actual: generated coast conditions relief, a region of that generated relief feeds intermediate, and its generated intermediate feeds meso. The last three images show the same 368.64 km regional zoom. There is no reset to Earth in the actual chain.'}
    links = ''.join(f'<a href="#{name}">{title}</a>' for name, title in cards)
    sections = ''.join(f'<section id="{name}"><h2>{title}</h2><p>{descriptions["complete" if name.startswith("complete") else "global" if name.startswith("paleo") else "structure" if name.endswith("structure") else "selected"]}</p><a href="{name}.png"><img src="{name}.png" alt="{title}"></a></section>' for name, title in cards)
    header = '''<h1>Terrain round trips: ideal versus actual</h1>
<p>Read each row from left to right. <strong>Ideal</strong> replaces learned predictions with the fields prescribed by the training objective. <strong>Actual</strong> passes each sampled output into the next stage. There are no independent Earth resets between generated stages.</p>
<p>The first two examples show the complete coast → relief → intermediate → meso generation. The ideal row marks the change from PaleoDEM to modern Earth training targets, because there is no single fully paired training planet across this boundary. The regional examples then hold the initial guide fixed to isolate the finer-stage round trip.</p>
<p>Each regional panel covers the same 368.64 km square. Full intermediate/structure context is 2,580.48 km. The first Earth panel is a reference used to form the coarse guide; its fine terrain is not supplied to generation. All samples use 32 steps, seed 42.</p>
<p><a href="README.md">Explanation of targets and handoffs</a> · <a href="provenance.json">Provenance and chain checks</a></p>'''
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Terrain round trips</title><style>'
        'body{font:16px system-ui;margin:0;background:#f4f4f4;color:#202c35}header{padding:24px 30px;background:white}p{max-width:1250px;line-height:1.5}'
        'nav{position:sticky;top:0;display:flex;gap:16px;flex-wrap:wrap;padding:12px 30px;background:white;border-bottom:1px solid #ccc;z-index:1}'
        'a{color:#245d7f}section{padding:24px 30px;scroll-margin-top:95px}img{width:100%;max-width:2200px}h2{font-size:24px}</style></head><body>'
        '<header>'+header+'</header><nav>'+links+'</nav>'+sections+'</body></html>', encoding='utf-8')
    md = '''# Terrain round trips: ideal versus actual

Read each paired flow left to right. The ideal line replaces each learned result with the training target; the actual line passes generated output into the next model. These are target-oracle illustrations, not claims that a conditional generator should copy a particular Earth range exactly.

Coast/relief train on PaleoDEMs, while the finer models train on modern Earth. There is no fully paired fine-resolution Paleo target. The two complete pipeline examples mark this source change in the ideal training-example row. Their actual row is one genuinely connected generated world, with bounded regional intermediate/meso sampling. These actual intermediate inputs are sampled directly from the generated relief cube, not an Earth guide or the pre-existing full planet. Separate matched Earth round trips hold the regional input fixed to make accumulated downstream changes visible.

The Earth references use the Alps, California and Great Plains. They are geographic review windows, not literal stored training records. We apply the saved training recipes to those windows. The intermediate source uses the same prefiltered ETOPO raster, spherical quadrature and half-precision target encoding as preparation. The meso source uses the original ETOPO DEM, the same bilinear lookup and three-sample area averaging as its preparation.

Selected pipeline: the intermediate ideal target is original Earth encoded terrain plus degraded-guide baseline minus original-guide baseline. The meso ideal target uses its original Earth residual plus the incoming ideal intermediate guide. This is the actual residual-preservation rule, applied across the handoff; it is a target construction, not a unique supervised label for a particular generated guide. No guide jitter is added during these controlled inference comparisons. The coarse guide is smoothed by one native coarse cell, within the trained smoothing range.

Experimental structure pipeline: the ideal 40 km field is the constrained Earth target. Its handoff uses target intermediate short-scale detail; the actual handoff uses sampled intermediate detail. Both use the same filtering/projection. Ideal meso uses the Earth residual rebased onto the ideal handoff and then the same smooth conservation operation as the actual final output. That final conservation operation is not part of the meso model's learned training target. Raw fields and the unprojected meso training target are saved in the regional NPZ files. This pilot remains unselected.

Regional panels all cover 368.64 km, north-up, at the same physical footprint. A 40 km field is resampled only for display onto the handoff grid. Full intermediate/structure model patches cover 2,580.48 km; meso generation uses 128 cells (368.64 km), although the adapted model trains on 80-cell crops. All metrical plots use the same colours, light and shading. Sampling is 32 steps, seed 42 per stage and region, independent local patches without production globe tiling/blending. No full planet meso run or training was launched.

The first Earth reference image is not a fine-resolution model input. Only its coarse guide reaches the actual regional chain. NPZ files save the exact inputs and outputs. The actual meso conditions are checked for exact equality to the preceding intermediate/handoff output crop, and actual relief is conditioned on its sampled coastline. Provenance includes checkpoint hashes.

'''
    for name, title in cards: md += f'## {title}\n\n![{title}]({name}.png)\n\n'
    (out/'README.md').write_text(md, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='outputs/terrain-round-trips')
    p.add_argument('--reuse-samples', action='store_true', help='Re-render saved flows and add any missing complete-chain regional tests')
    a = p.parse_args()
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()) and not a.reuse_samples: raise ValueError('Choose a fresh output folder')
    if not torch.cuda.is_available(): raise ValueError('Use CUDA for these bounded tests')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source = json.loads((ROOT/'outputs/model-training-examples/provenance.json').read_text())
    metadata = json.loads((out/'provenance.json').read_text()) if a.reuse_samples else dict(models={k: source['models'][k] for k in ['coast', 'relief', 'intermediate-selected', 'meso-selected', 'structure-40']},
        regions=[], global_prefixes=[], sampler_steps=32, seed=42, full_planet_meso=False, training_launched=False,
        regional_display_width_m=368640., intermediate_context_width_m=2580480.,
        ideal='training-target oracle, not deterministic reconstruction demand', meso_target='original Earth encoded residual rebased onto incoming ideal stage',
        constraint='smooth-v1 in experimental structure branch only')
    for model in metadata['models'].values():
        if digest(model['checkpoint']) != model['sha256']: raise ValueError('Checkpoint changed')
    start = time.perf_counter()
    if a.reuse_samples:
        other_cards = redraw_saved(out, metadata)
    else:
        other_cards = globals_(out, metadata)+regions(out, metadata)
    cards = complete_flows(out, metadata)+other_cards
    gallery(out, cards)
    metadata['last_operation_elapsed_seconds'] = time.perf_counter()-start
    (out/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Completed {len(cards)} paired flows in {metadata["last_operation_elapsed_seconds"]:.1f}s: {out}', flush=True)


if __name__ == '__main__': main()
