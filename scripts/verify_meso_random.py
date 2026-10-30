"""Run the real UI/GPU in random and mixed modes and retain review artifacts."""
from datetime import datetime
import json
from pathlib import Path
import tkinter as tk

import numpy as np
import rasterio

from scripts.verify_meso_ui import wait_until, screenshot
from toy_meso import gui
from toy_meso.conditioning import make_source


def main():
    base = gui.ROOT / 'outputs/meso/ui-random-verification'
    run = base / datetime.now().strftime('live-%Y%m%d-%H%M%S')
    run.mkdir(parents=True)
    gui.configure_tk()
    root = tk.Tk()
    root.withdraw()
    app = gui.Launcher(root)
    errors = []
    gui.messagebox.showerror = lambda title, message: errors.append(message)
    reports = []
    try:
        for source in ('Random terrain', 'Map + random'):
            folder = run / ('random' if source == 'Random terrain' else 'mixed')
            values = dict(gui.default_settings(), source=source, map_percent='75', folder=str(folder),
                          seed='42', scope='Region', west='5', south='44', east='7', north='46',
                          width='64', height='64', steps='8', batch='1', device='cuda')
            if source == 'Random terrain':
                values['dem'] = ''
            for key, value in values.items():
                app.variables[key].set(value)
            app.start()
            wait_until(root, lambda: not app.busy, f'{source} generation', timeout=180)
            assert app.status.get().startswith('Complete:'), (app.log.get('1.0', 'end'), errors)
            with rasterio.open(folder / 'region.tif') as dataset:
                first = dataset.read(1)
                assert dataset.shape == (64, 64)
                assert np.isfinite(first).all() and np.ptp(first) > 0
                assert dataset.tags()['input_source'] == ('random' if source == 'Random terrain' else 'mixed')
            assert app.preview.is_file()
            restored = gui.load_settings(folder / 'meso-launch.json')
            assert restored == values
            tiles = [path for path in (folder / 'tiles').rglob('*.npy') if '.tmp.' not in path.name]
            times = [path.stat().st_mtime_ns for path in tiles]
            app.variables['name'].set('Repeat.tif')
            app.start()
            wait_until(root, lambda: not app.busy, f'{source} cached export')
            assert app.status.get().startswith('Complete:'), app.log.get('1.0', 'end')
            with rasterio.open(folder / 'Repeat.tif') as dataset:
                np.testing.assert_array_equal(dataset.read(1), first)
            assert [path.stat().st_mtime_ns for path in tiles] == times
            report = dict(source=source, output=str(folder / 'region.tif'), tiles=len(tiles),
                          cached_export_identical=True, min_metres=float(first.min()), max_metres=float(first.max()))
            reports.append(report)
            print(f'GPU UI verified: {source}; {len(tiles)} cached tiles; identical repeat', flush=True)
        assert not errors, errors
        latitude = 90 - (np.arange(512)[:, None] + .5) * 180 / 512
        longitude = -180 + (np.arange(1024)[None, :] + .5) * 360 / 1024
        path = gui.default_settings()['dem']
        guides = [make_source(source, 42, path, -6000, 6000, map_weight=.75).sample(latitude, longitude)
                  for source in ('map', 'random', 'mixed')]
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
        for axis, field, title in zip(axes, guides, ('Input map', 'Random terrain — seed 42', '75% map + 25% random')):
            artist = axis.imshow(field, cmap='terrain', vmin=-6000, vmax=6000, extent=(-180, 180, -90, 90))
            axis.set_title(title)
            axis.set_xlabel('Longitude')
            axis.set_ylabel('Latitude')
        figure.colorbar(artist, ax=axes, label='Elevation input (metres)', shrink=.8)
        figure.suptitle('Terrain guides before the model adds detail')
        figure.savefig(base / 'input-guides.png', dpi=120)
        plt.close(figure)
        (base / 'verification.json').write_text(json.dumps(dict(checkpoint=values['checkpoint'], device='cuda',
                                                              steps=8, batch=1, runs=reports, errors=errors), indent=2))
        for key, value in gui.default_settings().items():
            app.variables[key].set(value)
        app.variables['source'].set('Map + random')
        app.variables['map_percent'].set('75')
        app.status.set('Ready. Choose an input source, then generate a globe or region.')
        app.log.configure(state='normal')
        app.log.delete('1.0', 'end')
        app.log.configure(state='disabled')
        app.progress.configure(mode='determinate', value=0)
        app.preview = None
        app.preview_button.configure(state='disabled')
        root.attributes('-topmost', True)
        root.deiconify()
        screenshot(root, base / 'ui-mixed.png')
        print(f'Random and mixed verification complete: {run}', flush=True)
    finally:
        if app.busy:
            app.cancel()
            wait_until(root, lambda: not app.busy, 'cleanup')
        app.close()


if __name__ == '__main__':
    main()
