"""Field-by-field comparison of two runs' snapshots (identity checks between sweeps; NaN-aware).

  python tools\\compare_runs.py <seed folder A> <seed folder B>
  python tools\\compare_runs.py <combination folder A> <combination folder B>   (every seedNNNN present in both)

Prints, per seed, the snapshots compared, the fields that differ, and the stats.json keys that differ.
"""
import json
import sys
from pathlib import Path

import numpy as np


def compare(a, b):
    sa = sorted((a/'snapshots').glob('t*.npz')); sb = {x.name: x for x in (b/'snapshots').glob('t*.npz')}
    diff, n = [], 0
    for x in sa:
        if x.name not in sb:
            diff.append((x.name, 'missing in B')); continue
        za, zb = np.load(x), np.load(sb[x.name])
        for k in sorted(set(za.files) | set(zb.files)):
            n += 1
            if k not in za.files or k not in zb.files:
                diff.append((x.name, k+' (only one side)')); continue
            u, v = za[k], zb[k]
            if u.shape != v.shape or not np.array_equal(u, v, equal_nan=u.dtype.kind == 'f'):
                diff.append((x.name, k))
    ja = json.loads((a/'stats.json').read_text()); jb = json.loads((b/'stats.json').read_text())
    keys = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
    print(f"{a.name}: {len(sa)} snapshots, {n} fields, {len(diff)} differ{': '+', '.join(f'{t} {k}' for t, k in diff[:8]) if diff else ''}; "
          f"stats.json keys that differ: {keys}")
    return not diff


def main(argv):
    A, B = Path(argv[0]), Path(argv[1])
    pairs = [(A, B)] if (A/'snapshots').is_dir() else [(A/s.name, B/s.name) for s in sorted(A.glob('seed*')) if (B/s.name).is_dir()]
    ok = all([compare(a, b) for a, b in pairs])
    print('IDENTICAL' if ok else 'DIFFERENT', f'({len(pairs)} seeds)')


if __name__ == '__main__':
    main(sys.argv[1:])
