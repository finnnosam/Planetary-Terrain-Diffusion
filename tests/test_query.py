"""Persistent regional requests must reuse completed guides and predictions."""
import json

import numpy as np
import pytest
import rasterio

from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.cli import main
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.gui import build_query_command
from planet_diffusion.query import query_region
from planet_diffusion.storage import save_state


class CountingBackend(DiagnosticBackend):
    def __init__(self):
        self.calls = {"coarse": 0, "latent": 0, "decoder": 0}

    def coarse_predict(self, *args):
        self.calls["coarse"] += 1
        return super().coarse_predict(*args)

    def predict(self, stage, *args):
        self.calls[stage] += 1
        return super().predict(stage, *args)


def test_query_new_regions_from_one_saved_run(tmp_path, monkeypatch):
    backend = CountingBackend()
    checkpoints = tmp_path / "state.checkpoints"
    first, metadata = generate_cube(backend, 42, face_coarse=2, coarse_steps=2,
                                    region=((-10, -5, 10, 5), 8, 4),
                                    checkpoint_dir=checkpoints, progress=lambda _: None)
    metadata["radius_metres"] = 6371000.
    state = tmp_path / "state"
    save_state(state, first, metadata)
    command, folder, output = build_query_command(
        tmp_path, (170, -5, -170, 5), 8, 4, stamp="fixture")
    assert folder == tmp_path
    assert output == tmp_path / "region-fixture.tif"
    assert command[command.index("--state") + 1] == str(state)
    assert command[command.index("--bounds") + 1:command.index("--bounds") + 5] == ["170", "-5", "-170", "5"]
    before = backend.calls.copy()

    original, original_info, _ = query_region(
        backend, state, (-10, -5, 10, 5), 8, 4, progress=lambda _: None)
    np.testing.assert_array_equal(original, first)
    assert original_info["regional_execution"]["decoder_model_calls"] == 0
    assert backend.calls == before

    bounds = (170, -5, -170, 5)
    a, info, _ = query_region(backend, state, bounds, 8, 4, progress=lambda _: None)
    assert backend.calls["coarse"] == before["coarse"]
    assert backend.calls["latent"] == before["latent"]
    assert backend.calls["decoder"] > before["decoder"]
    assert info["bounds"] == [170., -5., 190., 5.]

    calls = backend.calls.copy()
    b, resumed, _ = query_region(backend, state, bounds, 8, 4, progress=lambda _: None)
    np.testing.assert_array_equal(a, b)
    assert backend.calls == calls
    assert resumed["regional_execution"]["decoder_model_calls"] == 0
    assert resumed["regional_execution"]["prediction_disk_reads"] > 0

    output = tmp_path / "queried.tif"
    from planet_diffusion import query
    load_calls = []
    original_load = query.load_query
    def counted_load(*args, **kwargs):
        load_calls.append(1)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(query,"load_query",counted_load)
    assert main(["query", "--state", str(state), "--bounds", "170", "-5", "-170", "5",
                 "--width", "8", "--height", "4", "--output", str(output)]) == 0
    assert len(load_calls) == 1
    with rasterio.open(output) as dataset:
        np.testing.assert_array_equal(dataset.read(1), a)
        assert tuple(dataset.bounds) == (170, -5, 190, 5)


def test_query_rejects_mismatched_guides(tmp_path):
    backend = DiagnosticBackend()
    a, metadata = generate_cube(backend, 42, face_coarse=2, coarse_steps=2,
                                checkpoint_dir=tmp_path / "state.checkpoints",
                                progress=lambda _: None)
    metadata["radius_metres"] = 6371000.
    save_state(tmp_path / "state", a, metadata)
    manifest = tmp_path / "state.checkpoints" / "identity.json"
    identity = json.loads(manifest.read_text())
    identity["seed"] = 43
    manifest.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="different generation identities"):
        query_region(backend, tmp_path / "state", (-5, -5, 5, 5), 4, 4,
                     progress=lambda _: None)


def test_query_from_global_state_exports_saved_climate(tmp_path):
    backend = CountingBackend()
    elevation, metadata, climate = generate_cube(
        backend, 42, face_coarse=2, coarse_steps=2,
        checkpoint_dir=tmp_path / "state.checkpoints", with_climate=True,
        progress=lambda _: None)
    metadata["radius_metres"] = 6371000.
    save_state(tmp_path / "state", elevation, metadata, climate=climate)
    output = tmp_path / "regional.tif"
    climate_output = tmp_path / "regional-climate.tif"
    assert main(["query", "--state", str(tmp_path / "state"),
                 "--bounds", "-5", "-5", "5", "5", "--width", "4", "--height", "4",
                 "--output", str(output), "--climate-output", str(climate_output)]) == 0
    with rasterio.open(output) as heightmap, rasterio.open(climate_output) as climate_map:
        assert heightmap.shape == climate_map.shape == (4, 4)
        assert climate_map.count == 5
        assert np.isfinite(heightmap.read()).all()
        assert np.isfinite(climate_map.read()).all()
