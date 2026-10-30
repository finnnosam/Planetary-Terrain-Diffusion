"""Exercise the real UI worker with the user's checkpoint, including Stop/Resume."""
from datetime import datetime
import json
from pathlib import Path
import time
import tkinter as tk

import numpy as np
from PIL import ImageGrab
import rasterio

from toy_meso import gui


def wait_until(root, predicate, label, timeout=180):
    started = time.monotonic()
    while not predicate():
        root.update()
        if time.monotonic() - started > timeout:
            raise RuntimeError(f'Timed out: {label}')
        time.sleep(.03)
    root.update()


def screenshot(root, path):
    root.update()
    time.sleep(.5)
    root.update()
    ImageGrab.grab(bbox=(root.winfo_rootx(), root.winfo_rooty(),
                        root.winfo_rootx() + root.winfo_width(),
                        root.winfo_rooty() + root.winfo_height())).save(path)


def main():
    base = gui.ROOT / 'outputs/meso/ui-verification'
    folder = base / datetime.now().strftime('live-%Y%m%d-%H%M%S')
    gui.configure_tk()
    root = tk.Tk()
    root.withdraw()
    app = gui.Launcher(root)
    errors = []
    gui.messagebox.showerror = lambda title, message: errors.append(message)
    try:
        settings = dict(gui.default_settings(), folder=str(folder), scope='Region',
                        west='5', south='44', east='7', north='46',
                        width='64', height='64', steps='8', batch='1', device='cuda')
        for key, value in settings.items():
            app.variables[key].set(value)
        app.start()
        wait_until(root, lambda: 'tiles 1/' in app.log.get('1.0', 'end') or not app.busy,
                   'first tile completed')
        assert app.busy, (app.status.get(), app.log.get('1.0', 'end'), errors)
        app.cancel()
        wait_until(root, lambda: not app.busy, 'Stop completed')
        assert app.status.get().startswith('Stopped.'), app.status.get()
        assert not (folder / 'region.tif').exists()
        tiles = list((folder / 'tiles').rglob('*.npy'))
        tiles = [path for path in tiles if '.tmp.' not in path.name]
        assert tiles
        for path in tiles:
            assert np.isfinite(np.load(path)).all()
        saved_times = {str(path): path.stat().st_mtime_ns for path in tiles}
        print(f'Stop verified: {len(tiles)} finished tiles retained', flush=True)
        app.start()
        wait_until(root, lambda: not app.busy, 'Resume completed')
        assert app.status.get().startswith('Complete:'), (app.status.get(), app.log.get('1.0', 'end'), errors)
        assert all(Path(path).stat().st_mtime_ns == stamp for path, stamp in saved_times.items())
        with rasterio.open(folder / 'region.tif') as source:
            first = source.read(1)
            assert source.shape == (64, 64)
            assert source.tags()['version'] == gui.SAMPLER_VERSION
            assert np.isfinite(first).all()
            assert np.ptp(first) > 0
        assert app.preview.exists()
        print('Resume verified: completed elevation TIFF and preview', flush=True)
        restored = gui.load_settings(folder / 'meso-launch.json')
        assert restored == settings
        app.variables['name'].set('Repeat.tif')
        app.start()
        wait_until(root, lambda: not app.busy, 'Cached export completed')
        assert app.status.get().startswith('Complete:'), app.log.get('1.0', 'end')
        with rasterio.open(folder / 'Repeat.tif') as source:
            np.testing.assert_array_equal(source.read(1), first)
        assert not errors, errors
        print('Cache reuse verified: second export is identical', flush=True)
        report = dict(checkpoint=settings['checkpoint'], input=settings['dem'], device='cuda',
                      scope='Region', bounds=[5, 44, 7, 46], steps=8, batch=1,
                      retained_tiles=len(tiles), stop_resume=True, identical_cached_export=True,
                      output=str(folder / 'region.tif'), errors=errors)
        (base / 'verification.json').write_text(json.dumps(report, indent=2))
        for key, value in gui.default_settings().items():
            app.variables[key].set(value)
        app.preview = None
        app.preview_button.configure(state='disabled')
        app.log.configure(state='normal')
        app.log.delete('1.0', 'end')
        app.log.configure(state='disabled')
        app.status.set('Ready. Choose a map, then generate a globe or region.')
        app.progress.configure(mode='determinate', value=0)
        app.root.attributes('-topmost', True)
        root.deiconify()
        screenshot(root, base / 'ui-basic.png')
        app.toggle_advanced()
        screenshot(root, base / 'ui-advanced.png')
        print(f'UI verification complete: {folder}', flush=True)
    finally:
        if app.busy:
            app.cancel()
            wait_until(root, lambda: not app.busy, 'cleanup')
        app.close()


if __name__ == '__main__':
    main()
