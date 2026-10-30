"""Pickle a PlateSim (its flexure keeps a module reference: pickled by name). Resume is bit-identical (tested 28 Oct)."""
import copyreg, importlib, pickle, types
copyreg.pickle(types.ModuleType, lambda m: (importlib.import_module, (m.__name__,)))
def save(sim, f):
    with open(f, 'wb') as fh: pickle.dump(sim, fh, protocol=5)
def load(f):
    with open(f, 'rb') as fh: return pickle.load(fh)
