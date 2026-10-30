"""Training alignment, geographic placement, topology and cache regression checks."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from planet_diffusion import cube
from toy_meso import planet_tiled as tiled
from toy_meso.common import Normalizer
from toy_meso.model import upsample_condition
from toy_meso.sphere import encode


def test_tile_condition_matches_training_interpolation_including_halos():
    grid = tiled.Grid(64, 16, 8, factor=4)
    n = grid.N // grid.factor
    side = n + 1 + 2 * grid.coarse_halo
    cells = np.random.default_rng(12).normal(size=(1, 6, side, side)).astype(np.float32)
    for key in grid.all_tiles():
        f, ky, kx = key
        cy = grid.starts[ky] // grid.factor - 1 + grid.coarse_halo
        cx = grid.starts[kx] // grid.factor - 1 + grid.coarse_halo
        count = grid.size // grid.factor + 2
        patch = torch.from_numpy(cells[:, f, cy:cy + count, cx:cx + count])[:, None]
        expected = upsample_condition(patch, grid.factor)[0, 0].numpy()
        np.testing.assert_allclose(tiled.tile_condition(cells, grid.factor, grid, *key),
                                   expected, atol=3e-7)


def test_coarse_averages_are_centred_on_the_original_fine_cells():
    class LongitudeRaster:
        def sample(self, lat, lon):
            return lon.astype(np.float32)

    grid = tiled.Grid(64, 16, 8, factor=4)
    n = grid.N // grid.factor
    coarse = tiled.coarse_condition_cells(LongitudeRaster(), n, grid.factor, grid.coarse_halo)
    # Longitude is linear in face 0's x chart. Its mean should therefore be at
    # the block centre, rather than at its first fine node or a shifted output.
    x = np.arange(n + 1)
    expected = -45 + 90 * (x + (grid.factor - 1) / (2 * grid.factor)) / n
    row = coarse[0, 0, grid.coarse_halo + n // 2,
                 grid.coarse_halo:grid.coarse_halo + n + 1]
    np.testing.assert_allclose(row, expected, atol=5e-6)
    up = tiled.condition_at(coarse, grid.factor, grid, 0,
                            np.full(grid.N + 1, grid.N // 2), np.arange(grid.N + 1))
    np.testing.assert_allclose(up, -45 + 90 * np.arange(grid.N + 1) / grid.N, atol=5e-6)


def test_full_predictions_preserve_geographic_positions_and_shared_edges(tmp_path):
    grid = tiled.Grid(64, 16, 8, factor=4)
    cache = tiled.TileCache(tmp_path)
    for f, ky, kx in grid.all_tiles():
        y = grid.starts[ky] + np.arange(grid.size)[:, None]
        x = grid.starts[kx] + np.arange(grid.size)[None, :]
        p = cube.directions(f, y, x, grid.N)
        cache.save((f, ky, kx), (20 * p[..., 0] + 40 * p[..., 1] + 60 * p[..., 2]).astype(np.float32))
    sums, weights = tiled.blend(grid.all_tiles(), grid, cache)
    result = sums / weights
    expected = np.stack([20 * (p := cube.directions(f, np.arange(grid.N + 1)[:, None],
                                                    np.arange(grid.N + 1)[None, :], grid.N))[..., 0]
                         + 40 * p[..., 1] + 60 * p[..., 2] for f in range(6)])[None]
    np.testing.assert_allclose(result, expected, atol=2e-5)
    flat = result.reshape(-1)
    members, ids, owners, _ = cube.edge_groups(grid.N)
    np.testing.assert_array_equal(flat[members], flat[owners[ids]])
    # Include corners, face edges and tile overlap positions in a partial request.
    needed = grid.group_members(grid.flat(np.array([0, 1, 4, 5]),
                                          np.array([0, 8, 32, 64]),
                                          np.array([0, 32, 64, 24])))
    keys = grid.covering_tiles(needed)
    small_sums, small_weights = tiled.blend(keys, grid, cache)
    np.testing.assert_array_equal(small_sums.reshape(-1)[needed], sums.reshape(-1)[needed])
    np.testing.assert_array_equal(small_weights.reshape(-1)[needed], weights.reshape(-1)[needed])


def test_generation_caches_encoded_elevations_not_normalized_residuals(tmp_path):
    class ZeroDenoiser(torch.nn.Module):
        def __call__(self, x, sigma, cond):
            return torch.zeros_like(x)

    grid = tiled.Grid(64, 16, 8, factor=4)
    side = grid.N // grid.factor + 1 + 2 * grid.coarse_halo
    condition = np.full((1, 6, side, side), 25, np.float32)
    norm = Normalizer(dict(cond_mean=3, cond_std=2, res_std=4), grid.factor)
    cache = tiled.TileCache(tmp_path)
    key = (0, 3, 3)
    tiled.compute_tiles([key], ZeroDenoiser(), norm, condition, grid.factor, grid,
                        0, cache, 8, 2, 'cpu')
    np.testing.assert_array_equal(cache.load(key), np.full((16, 16), 25, np.float32))


def test_region_matches_full_export_and_old_caches_are_rejected(tmp_path, monkeypatch):
    import rasterio
    from PIL import Image

    class ZeroDenoiser(torch.nn.Module):
        def __call__(self, x, sigma, cond):
            return torch.zeros_like(x)

    run = tmp_path / 'run'
    run.mkdir()
    (run / 'latest.pt').write_bytes(b'test checkpoint')
    dem = tmp_path / 'input.png'
    Image.fromarray(np.tile(np.arange(128, dtype=np.uint8) + 100, (64, 1))).save(dem)
    out = tmp_path / 'output'
    h = np.pi * 6371000 / 2 / 64
    norm = Normalizer(dict(cond_mean=0, cond_std=1, res_std=1), 4)
    ck = dict(step=222000, meta=dict(spacing_m=h, factor=4, size=16))
    monkeypatch.setattr(tiled, 'load_checkpoint', lambda *args: (ZeroDenoiser(), norm, ck))
    common = ['--run', str(run), '--dem', str(dem), '--out', str(out),
              '--black-metres', '-6000', '--white-metres', '6000',
              '--steps', '8', '--batch', '2', '--device', 'cpu']
    tiled.main(common + ['--bounds', '-22.5', '-11.25', '22.5', '11.25',
                         '--width', '16', '--height', '8', '--name', 'region.tif'])
    tiled.main(common + ['--height', '64'])
    with rasterio.open(out / 'region.tif') as region, rasterio.open(out / 'planet.tif') as planet:
        np.testing.assert_array_equal(region.read(1), planet.read(1)[28:36, 56:72])
        assert region.tags()['version'] == tiled.VERSION
        assert np.ptp(region.read(1)) > 0
    settings_path = out / 'settings.json'
    settings = json.loads(settings_path.read_text())
    settings['version'] = 'meso-tiled-v1'
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(SystemExit, match='different settings'):
        tiled.main(common + ['--height', '64', '--name', 'old-cache.tif'])
    settings['version'] = tiled.VERSION
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(SystemExit, match='batch'):
        tiled.main(common + ['--batch', '4', '--height', '64', '--name', 'different-batch.tif'])


@pytest.mark.parametrize('stride', [0, -8, 7, 16, 24])
def test_invalid_tile_strides_fail_before_generation(stride):
    with pytest.raises(ValueError, match='stride'):
        tiled.Grid(64, 16, stride, factor=4)


@pytest.mark.parametrize('source', ['random', 'mixed'])
def test_random_and_mixed_regions_match_globe_and_cache_pins_source(tmp_path, monkeypatch, source):
    import rasterio
    from PIL import Image

    class ZeroDenoiser(torch.nn.Module):
        def __call__(self, x, sigma, cond):
            return torch.zeros_like(x)

    run = tmp_path / 'run'
    run.mkdir()
    (run / 'latest.pt').write_bytes(b'checkpoint')
    dem = tmp_path / 'map.png'
    Image.new('L', (32, 16), 180).save(dem)
    out = tmp_path / 'output'
    norm = Normalizer(dict(cond_mean=0, cond_std=1, res_std=1), 4)
    ck = dict(step=1, meta=dict(spacing_m=np.pi * 6371000 / 2 / 64, factor=4, size=16))
    monkeypatch.setattr(tiled, 'load_checkpoint', lambda *args: (ZeroDenoiser(), norm, ck))
    common = ['--run', str(run), '--out', str(out), '--source', source,
              '--steps', '8', '--batch', '2', '--device', 'cpu', '--seed', '42']
    if source == 'mixed':
        common += ['--dem', str(dem), '--map-weight', '.75']
    tiled.main(common + ['--bounds', '-22.5', '-11.25', '22.5', '11.25', '--width', '16', '--height', '8'])
    tiled.main(common + ['--height', '64'])
    with rasterio.open(out / 'region.tif') as region, rasterio.open(out / 'planet.tif') as planet:
        np.testing.assert_array_equal(region.read(1), planet.read(1)[28:36, 56:72])
        assert planet.tags()['input_source'] == source
    with pytest.raises(SystemExit, match='different settings'):
        tiled.main(common + ['--seed', '43', '--height', '8', '--name', 'changed-seed.tif'])
    if source == 'mixed':
        with pytest.raises(SystemExit, match='map_weight'):
            tiled.main(common + ['--map-weight', '.5', '--height', '8', '--name', 'changed-mix.tif'])
        with pytest.raises(SystemExit, match='input_source'):
            tiled.main(common + ['--source', 'map', '--height', '8', '--name', 'changed-source.tif'])


@pytest.mark.parametrize('args', [['--source', 'mixed'], ['--source', 'map'],
                                 ['--source', 'random', '--map-weight', 'nan'],
                                 ['--source', 'random', '--map-weight', '1.1']])
def test_invalid_sources_fail_before_model_loading(tmp_path, monkeypatch, args):
    def unexpected(*args):
        pytest.fail('invalid input should not load the model')
    monkeypatch.setattr(tiled, 'load_checkpoint', unexpected)
    with pytest.raises(SystemExit):
        tiled.main(['--run', str(tmp_path), '--out', str(tmp_path / 'out')] + args)
    assert not (tmp_path / 'out').exists()


def test_interrupted_array_write_does_not_expose_partial_cache(tmp_path, monkeypatch):
    target = tmp_path / 'coarse.npy'
    original = np.save

    def interrupted(path, values):
        original(path, values)
        raise KeyboardInterrupt('stopped before publishing')

    monkeypatch.setattr(tiled.np, 'save', interrupted)
    with pytest.raises(KeyboardInterrupt):
        tiled.save_array(target, np.ones((4, 4)))
    assert not target.exists()
    monkeypatch.setattr(tiled.np, 'save', original)
    tiled.save_array(target, np.full((4, 4), 2))
    np.testing.assert_array_equal(np.load(target), np.full((4, 4), 2))


def test_partial_coarse_cache_matches_complete_averages_and_expands(tmp_path):
    class Raster:
        def sample(self, lat, lon):
            return (30 * np.sin(np.radians(lat)) + 50 * np.cos(np.radians(lon))).astype(np.float32)

    grid = tiled.Grid(64, 16, 8, factor=4)
    complete = tiled.coarse_condition_cells(Raster(), 16, 4, grid.coarse_halo)
    keys = [(0, 3, 3)]
    partial = tiled.ensure_coarse_condition(tmp_path, grid, keys, Raster)
    valid = np.load(tmp_path / 'coarse-valid.npy')
    assert valid.sum() < valid.size / 20
    np.testing.assert_array_equal(partial[0][valid], complete[0][valid])
    for key in keys:
        np.testing.assert_array_equal(tiled.tile_condition(partial, 4, grid, *key),
                                      tiled.tile_condition(complete, 4, grid, *key))

    def unnecessary():
        pytest.fail('cached inputs should not reload or sample the input map')

    tiled.ensure_coarse_condition(tmp_path, grid, keys, unnecessary)
    keys += [(1, 0, 0), (4, 8, 8)]
    expanded = tiled.ensure_coarse_condition(tmp_path, grid, keys, Raster)
    expanded_valid = np.load(tmp_path / 'coarse-valid.npy')
    assert np.all(expanded_valid[valid])
    assert expanded_valid.sum() > valid.sum()
    np.testing.assert_array_equal(expanded[0][expanded_valid], complete[0][expanded_valid])
    for key in keys:
        np.testing.assert_array_equal(tiled.tile_condition(expanded, 4, grid, *key),
                                      tiled.tile_condition(complete, 4, grid, *key))


def test_sparse_blending_and_sampling_match_dense_across_seams_and_poles(tmp_path):
    grid = tiled.Grid(64, 16, 8, factor=4)
    cache = tiled.TileCache(tmp_path)
    rng = np.random.default_rng(12)
    for key in grid.all_tiles():
        cache.save(key, rng.normal(0, 60, (grid.size, grid.size)))
    acc, weights = tiled.blend(grid.all_tiles(), grid, cache)
    dense = tiled.decode(acc / weights)
    points = rng.normal(size=(300, 3))
    points /= np.linalg.norm(points, axis=-1, keepdims=True)
    points = np.concatenate((points, np.array([[0, 0, 1], [0, 0, -1], [1, 1, 0],
                                              [-1, 0, 0], [1, 1, 1]], float)))
    face, y, x = cube.coordinates(points, grid.N)
    iy, ix = np.floor(y).astype(int), np.floor(x).astype(int)
    needed = grid.group_members(np.concatenate([grid.flat(face, np.minimum(iy + dy, grid.N),
                                                          np.minimum(ix + dx, grid.N))
                                                for dy in (0, 1) for dx in (0, 1)]))
    keys = grid.covering_tiles(needed)
    sparse = tiled.blend_region(keys, grid, cache, needed)
    np.testing.assert_array_equal(sparse, dense.reshape(-1)[needed])
    np.testing.assert_array_equal(tiled.sample_region(needed, sparse, grid, points), cube.sample(dense, points)[0])


def test_partial_coarse_cache_interrupted_before_mask_publish_is_recomputed(tmp_path, monkeypatch):
    class Raster:
        calls = 0

        def sample(self, lat, lon):
            self.calls += 1
            return np.full(np.broadcast_shapes(lat.shape, lon.shape), 80, np.float32)

    grid = tiled.Grid(64, 16, 8, factor=4)
    raster = Raster()
    save = tiled.save_array

    def interrupted(path, values):
        if Path(path).name == 'coarse-valid.npy' and values.any():
            raise KeyboardInterrupt('stopped after input values, before validity mask')
        save(path, values)

    monkeypatch.setattr(tiled, 'save_array', interrupted)
    with pytest.raises(KeyboardInterrupt):
        tiled.ensure_coarse_condition(tmp_path, grid, [(0, 3, 3)], lambda: raster)
    assert (tmp_path / 'coarse.npy').exists()
    assert not np.load(tmp_path / 'coarse-valid.npy').any()
    first_calls = raster.calls
    monkeypatch.setattr(tiled, 'save_array', save)
    values = tiled.ensure_coarse_condition(tmp_path, grid, [(0, 3, 3)], lambda: raster)
    assert raster.calls > first_calls
    valid = np.load(tmp_path / 'coarse-valid.npy')
    assert valid.any()
    assert np.all(values[0][valid] == 80)


def test_legacy_complete_coarse_cache_needs_no_mask_or_source(tmp_path):
    grid = tiled.Grid(64, 16, 8, factor=4)
    side = 64 // 4 + 1 + 2 * grid.coarse_halo
    values = np.full((1, 6, side, side), 40, np.float32)
    np.save(tmp_path / 'coarse.npy', values)

    def unnecessary():
        pytest.fail('legacy complete input should be reused')

    np.testing.assert_array_equal(tiled.ensure_coarse_condition(tmp_path, grid, grid.all_tiles(), unnecessary), values)
    assert not (tmp_path / 'coarse-valid.npy').exists()


def test_cached_export_never_moves_the_model_to_gpu(tmp_path, monkeypatch):
    from PIL import Image

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            assert device == 'cpu', 'cached export must not use the GPU'
            return self

        def __call__(self, x, sigma, cond):
            return torch.zeros_like(x)

    run = tmp_path / 'run'
    run.mkdir()
    (run / 'latest.pt').write_bytes(b'checkpoint')
    dem = tmp_path / 'map.png'
    Image.new('L', (32, 16), 180).save(dem)
    network = Model()
    norm = Normalizer(dict(cond_mean=0, cond_std=1, res_std=1), 4)
    checkpoint = dict(step=1, meta=dict(spacing_m=np.pi * 6371000 / 2 / 64, factor=4, size=16))
    monkeypatch.setattr(tiled, 'load_checkpoint', lambda path, device: (network, norm, checkpoint))
    common = ['--run', str(run), '--out', str(tmp_path / 'out'), '--dem', str(dem),
              '--bounds', '0', '0', '1', '1', '--height', '8', '--width', '8', '--steps', '8', '--batch', '2']
    tiled.main(common + ['--device', 'cpu'])
    first_moves = network.moves[:]
    tiled.main(common + ['--device', 'cuda', '--name', 'cached.tif'])
    assert network.moves == first_moves


def test_interrupted_export_resumes_without_overwriting_finished_outputs(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import rasterio
    from PIL import Image

    class ZeroDenoiser(torch.nn.Module):
        def __call__(self, x, sigma, cond):
            return torch.zeros_like(x)

    run = tmp_path / 'run'
    run.mkdir()
    (run / 'latest.pt').write_bytes(b'checkpoint')
    dem = tmp_path / 'input.png'
    Image.new('L', (32, 16), 128).save(dem)
    out = tmp_path / 'output'
    norm = Normalizer(dict(cond_mean=0, cond_std=1, res_std=1), 4)
    ck = dict(step=1, meta=dict(spacing_m=np.pi * 6371000 / 2 / 64, factor=4, size=16))
    monkeypatch.setattr(tiled, 'load_checkpoint', lambda *args: (ZeroDenoiser(), norm, ck))
    command = ['--run', str(run), '--dem', str(dem), '--out', str(out), '--height', '8',
               '--steps', '8', '--batch', '2', '--device', 'cpu']
    original = plt.imsave

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('stopped during preview export')

    monkeypatch.setattr(plt, 'imsave', interrupted)
    out.mkdir()
    # A previously completed output may itself have .tmp in its name.
    previous = out / 'planet.tmp.tif'
    previous.write_bytes(b'keep this completed output')
    with pytest.raises(KeyboardInterrupt):
        tiled.main(command)
    assert not (out / 'planet.tif').exists()
    assert (out / '.incomplete/planet.tif').exists()
    assert previous.read_bytes() == b'keep this completed output'
    tiles = list((out / 'tiles').rglob('*.npy'))
    times = [path.stat().st_mtime_ns for path in tiles]
    monkeypatch.setattr(plt, 'imsave', original)
    tiled.main(command)
    assert [path.stat().st_mtime_ns for path in tiles] == times
    with rasterio.open(out / 'planet.tif') as dataset:
        assert dataset.shape == (8, 16)
        assert np.isfinite(dataset.read(1)).all()
    assert (out / 'planet.preview.png').exists()
    assert not (out / '.incomplete/planet.tif').exists()
    assert previous.read_bytes() == b'keep this completed output'
