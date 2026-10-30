"""Check full-size lazy input preparation against the old complete input."""
import json
import time

import numpy as np

from scripts.profile_meso_speed import OUT, ROOT
from toy_meso.planet_tiled import Grid, ensure_coarse_condition
from toy_meso.conditioning import make_source

reference = OUT / 'region-baseline-mixed'
settings = json.loads((reference / 'settings.json').read_text())
grid = Grid(settings['fine_intervals'], settings['tile'], settings['stride'], settings['factor'])
folder = OUT / 'full-condition-current'
folder.mkdir(exist_ok=True)
start = time.perf_counter()
values = ensure_coarse_condition(folder, grid, grid.all_tiles(),
    lambda: make_source('mixed', 42, ROOT / 'human_input/B2W.png', -6000, 6000, None, .75))
elapsed = time.perf_counter() - start
valid = np.load(folder / 'coarse-valid.npy')
result = dict(seconds=elapsed, required_cells=int(valid.sum()), total_cells=int(valid.size),
              used_input_identical=bool(np.array_equal(values[0][valid], np.load(reference / 'coarse.npy')[0][valid])))
(OUT / 'full-condition-current.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
