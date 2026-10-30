"""Parcel back-tracing through saved plate rotations (planet_layer.coupling)."""
import json
from pathlib import Path

import numpy as np
import pytest

from planet_layer import coupling, plates as P
from planet_layer.sphere_graph import CubeSphere


@pytest.fixture(scope='module')
def history():
    # a folder inside the project (pytest's own temp folder can be unreadable on Windows: WinError 5)
    import shutil
    d = Path(__file__).resolve().parents[1]/'outputs'/'test-tmp'/'coupling-snapshots'
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    g = CubeSphere(48)
    sim = P.PlateSim(g, seed=4, t0=-6., params={'erosion': False})
    for _ in range(6):
        sim.step()
        st = sim.state()
        np.savez(d/f't{sim.t:+07.1f}.npz', **{k: v for k, v in st.items() if k != 'time_myr'})
        (d/f't{sim.t:+07.1f}-plates.json').write_text(json.dumps(sim.plate_table()))
    # the start state too
    return coupling.PlateHistory(d), sim


def test_parcels_on_one_plate_follow_its_rotation_exactly(history):
    h, sim = history
    fine = CubeSphere(64)
    p0 = fine.xyz.copy()
    own0 = h.owner(0., p0)
    p = p0.copy()
    stayed = np.ones(fine.size, bool)
    for t in range(0, -5, -1):
        p_prev = coupling.step_back(h, float(t), p)
        stayed &= h.owner(float(t), p) == own0
        p = p_prev
    _, Q0 = h.fields(0.)
    _, Q5 = h.fields(-5.)
    checked = 0
    for k in np.unique(own0[stayed]):
        if k not in Q5:
            continue
        sel = stayed & (own0 == k)
        expect = p0[sel]@Q0[k]@Q5[k].T
        assert np.allclose(p[sel], expect, atol=1e-9)
        checked += sel.sum()
    assert checked > .5*fine.size


def test_smoothing_conserves_volume(history):
    h, _ = history
    v = np.zeros(h.g.size); v[123] = 1000.
    s = h.smooth(v, 80e3)
    assert abs((s*h.g.area).sum()-(v*h.g.area).sum()) < 1e-6*(v*h.g.area).sum()
    assert s.max() < 1000.


def test_ocean_forcing_telescopes_along_parcel_paths(history):
    """Where a parcel is never continental at both ends of an interval (it may touch a coast or a converting arc for
    one sample), start surface + summed surface forcing = the plate layer's present elevation, and it gets no load."""
    h, _ = history
    fine = CubeSphere(40)
    out = Path(__file__).resolve().parents[1]/'outputs'/'test-tmp'/'coupling-forcing'
    F = coupling.build_forcing(h, fine, 5, out, progress=lambda s: None)
    p = fine.xyz.copy()
    c = [h.sample(0., 'continental', p)]
    for t in range(0, -5, -1):
        p = coupling.step_back(h, float(t), p)
        c.append(h.sample(float(t-1), 'continental', p))
    c = np.array(c)
    ocean_path = ~((c[1:] > 0) & (c[:-1] > 0)).any(0)
    touched = ocean_path & (c > 0).any(0)
    z_end = np.asarray(F['z0_m'], np.float64)+np.asarray(F['surface_m'], np.float64).sum(0)
    z_plate = h.sample(0., 'elevation_m', fine.xyz)
    assert ocean_path.mean() > .3 and touched.sum() > 0
    assert np.abs(z_end-z_plate)[ocean_path].max() < .05      # float32 storage only
    assert np.all(np.asarray(F['load_m'])[:, ocean_path] == 0)
