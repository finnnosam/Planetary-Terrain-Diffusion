"""Exercise the real process exit code and approval/resume for random and mixed input."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/meso/random-diagnosis'
results = {}
for source in ('random', 'mixed'):
    folder = OUT / f'live-approval-{source}'
    cmd = [sys.executable, '-u', '-m', 'toy_meso.planet_tiled', '--run', 'outputs/meso/run2-lowlr',
           '--out', str(folder), '--source', source, '--seed', '4100897701', '--steps', '32',
           '--batch', '16', '--bounds', '5', '44', '7', '46', '--width', '64', '--height', '64',
           '--device', 'cuda', '--pause-for-input']
    if source == 'mixed':
        cmd += ['--dem', 'human_input/B2W.png', '--black-metres', '-6000', '--white-metres', '6000',
                '--map-weight', '.5']
    start = time.perf_counter()
    paused = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    preview_seconds = time.perf_counter() - start
    print(paused.stdout, flush=True)
    assert paused.returncode == 42, paused.stderr
    assert not (folder / 'tiles').exists() and not (folder / 'region.tif').exists()
    coarse_stamp = (folder / 'coarse.npy').stat().st_mtime_ns
    start = time.perf_counter()
    generated = subprocess.run(cmd + ['--approve-input'], cwd=ROOT, text=True, capture_output=True)
    generation_seconds = time.perf_counter() - start
    print(generated.stdout, flush=True)
    assert generated.returncode == 0, generated.stderr
    assert (folder / 'coarse.npy').stat().st_mtime_ns == coarse_stamp
    assert (folder / 'region.tif').exists()
    assert json.loads((folder / 'input-review.json').read_text())['approved']
    results[source] = dict(preview_exit_code=paused.returncode, preview_seconds=preview_seconds,
                           approved_exit_code=generated.returncode, approved_seconds=generation_seconds,
                           prepared_input_reused=True, tiles=len(list((folder / 'tiles').glob('f*/*.npy'))))
(OUT / 'live-approval-results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
