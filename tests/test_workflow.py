import numpy as np
import pytest
import rasterio
from planet_diffusion.backends import DiagnosticBackend
from planet_diffusion.storage import save_state, load_state, export_tiff, verify_state
from planet_diffusion.cli import main


@pytest.fixture(scope="module")
def planet():
    from planet_diffusion.cube_generate import generate_cube
    a, meta = generate_cube(DiagnosticBackend(), 17, 2, 2, progress=lambda _: None)
    meta["radius_metres"] = 1000000.
    return a, meta


def test_geotiff_and_overlapping_dateline_tiles(tmp_path, planet):
    a, meta = planet
    save_state(tmp_path/"state", a, meta)
    a, meta = load_state(tmp_path/"state")
    assert verify_state(a, meta)["passed"]
    export_tiff(tmp_path/"global.tif", a, meta, height=64)
    export_tiff(tmp_path/"tile.tif", a, meta, height=64, window=(120, 10, 20, 12))
    export_tiff(tmp_path/"detail.tif", a, meta, window=(1000, 10, 40, 12))
    with rasterio.open(tmp_path/"global.tif") as full, rasterio.open(tmp_path/"tile.tif") as tile:
        assert (full.width, full.height) == (128, 64)
        assert tuple(full.bounds) == (-180., -90., 180., 90.)
        assert full.dtypes == ("float32",)
        assert full.units == ("m",)
        assert full.crs.to_epsg() is None
        assert full.tags()["seed"] == "17"
        assert full.tags()["AREA_OR_POINT"] == "Area"
        assert full.scales == (1.,) and full.offsets == (0.,)
        expected = np.concatenate([full.read(1)[10:22, 120:], full.read(1)[10:22, :12]], axis=1)
        np.testing.assert_array_equal(tile.read(1), expected)
        assert tile.bounds.right > 180
    assert main(["verify", "--state", str(tmp_path/"state")]) == 0
    meta["elevation_sha256"] = "bad"
    assert not verify_state(a, meta)["passed"]


def test_cli_diagnostic(tmp_path):
    assert main(["generate", "--backend", "diagnostic", "--seed", "1", "--coarse-height", "4",
                 "--coarse-steps", "2", "--state", str(tmp_path/"s"), "--output", str(tmp_path/"map.tif")]) == 0
    assert main(["export", "--state", str(tmp_path/"s"), "--output", str(tmp_path/"tile.tif"),
                 "--window", "1000", "0", "40", "20"]) == 0


def test_invalid_export(tmp_path, planet):
    a, meta = planet
    with pytest.raises(ValueError):
        export_tiff(tmp_path/"bad.tif", a, meta, window=(0, -1, 10, 10))


def test_steep_seam_is_reported_separately_from_topology(tmp_path):
    a = np.zeros((65, 128), np.float32)
    a[8:56, 0] = 10000
    save_state(tmp_path/"state", a, {"native_height": 64})
    a, meta = load_state(tmp_path/"state")
    report = verify_state(a, meta)
    assert report["passed"]  # A continuous interpolant still exists.
    assert report["seam_statistical_outlier"]  # But the seam is suspiciously steep.
