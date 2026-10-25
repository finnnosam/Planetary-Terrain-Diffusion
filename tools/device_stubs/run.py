"""Plumbing harness for the Linux device shell, which has no numba, ducc0 or pytest (twenty-ninth pass): flexure is
local Airy and fields.random_field a cosine sum; numba kernels run as pure Python. Plumbing only, never for judging.

  python3 tools/device_stubs/run.py test_short_run_invariants test_split_partitions_a_plate_without_resampling

Runs the named tests of tests/test_plates.py (a monkeypatch argument gets a minimal stand-in); keep each call under
~170 s (n = 48: ~2 s per step). `import run` from another script in this folder applies the stubs."""
import sys, time, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stubs')]
import numpy as np
from planet_layer import fields, plates as P

class Airy:
    def __init__(self, rho_load, rho_mantle): self.r = rho_load/rho_mantle
    @classmethod
    def for_graph(cls, graph, te, rho_load=1., rho_mantle=3300., nthreads=1, **kw): return cls(rho_load, rho_mantle)
    def deflection(self, load): return np.asarray(load, np.float64)*self.r

def random_field(xyz, lmax, slope, rng):
    d = rng.normal(size=(12, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    k = rng.uniform(1, max(2., lmax/2), 12); ph = rng.uniform(0, 2*np.pi, 12); a = k**(-slope/2)
    f = (a*np.cos(np.pi*k*(xyz@d.T)+ph)).sum(1)
    return (f-f.mean())/f.std()

P.SphereFlexure = Airy
fields.random_field = random_field

class MP:
    def setattr(self, obj, name, val): setattr(obj, name, val)

if __name__ == "__main__":
  import tests.test_plates as T
  for name in sys.argv[1:]:
      t0 = time.time()
      fn = getattr(T, name)
      fn(MP()) if fn.__code__.co_argcount else fn()
      print(f'PASS {name} {time.time()-t0:.0f}s', flush=True)
