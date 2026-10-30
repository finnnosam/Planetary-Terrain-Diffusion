"""Random sources are global, repeatable, independent of query size, and mix in metres."""
import numpy as np
from PIL import Image

from toy_meso import conditioning
from toy_meso.planet_tiled import Grid, coarse_condition_cells


def coordinates():
    return np.linspace(-80, 80, 30)[:, None], np.linspace(-180, 180, 60)[None, :]


def test_seeded_random_globe_has_elevation_variation_and_uses_all_seed_bits():
    lat, lon = coordinates()
    first = conditioning.RandomElevation(0).sample(lat, lon)
    np.testing.assert_array_equal(first, conditioning.RandomElevation(0).sample(lat, lon))
    assert not np.array_equal(first, conditioning.RandomElevation(1).sample(lat, lon))
    assert not np.array_equal(first, conditioning.RandomElevation(2**32).sample(lat, lon))
    assert first.dtype == np.float32 and np.isfinite(first).all()
    assert np.min(first) < -1000 and np.max(first) > 1000
    assert .1 < np.mean(first > 0) < .7


def test_random_is_identical_for_regional_and_global_sampling_and_continuous_at_seams():
    source = conditioning.RandomElevation(42)
    lat, lon = coordinates()
    full = source.sample(lat, lon)
    np.testing.assert_array_equal(full[8:16, 20:30], source.sample(lat[8:16], lon[:, 20:30]))
    latitude = np.linspace(-90, 90, 50)
    np.testing.assert_array_equal(source.sample(latitude, -180), source.sample(latitude, 180))
    # Two distinct points ~22 cm apart can differ more with the finer spectrum;
    # exact -180/+180 coordinates above must still be bit-identical.
    np.testing.assert_allclose(source.sample(latitude, -179.999999), source.sample(latitude, 179.999999), atol=.05)
    for pole in (-90, 90):
        values = source.sample(pole, np.linspace(-180, 180, 100))
        np.testing.assert_array_equal(values, np.full_like(values, values[0]))


def test_mixed_opaque_map_endpoints_and_weight_are_in_metres(tmp_path):
    path = tmp_path / 'map.png'
    Image.fromarray(np.tile(np.arange(32, dtype=np.uint8) * 8, (16, 1))).save(path)
    lat, lon = coordinates()
    map_values = conditioning.make_source('map', 42, path, -6000, 6000).sample(lat, lon)
    random_values = conditioning.make_source('random', 42).sample(lat, lon)
    for weight in (0, .25, .5, 1):
        actual = conditioning.make_source('mixed', 42, path, -6000, 6000, map_weight=weight).sample(lat, lon)
        expected = weight * map_values + (1 - weight) * random_values
        np.testing.assert_array_equal(actual, expected)


def test_transparency_fills_random_and_hidden_pixels_do_not_bleed(tmp_path):
    pixels = np.full((8, 16, 4), 255, dtype=np.uint8)
    pixels[:, 8:, 3] = 0
    path = tmp_path / 'paint.png'
    Image.fromarray(pixels).save(path)
    lat, lon = coordinates()
    source = conditioning.make_source('mixed', 0, path, -6000, 6000, map_weight=1)
    actual = source.sample(lat, lon)
    random = conditioning.RandomElevation(0).sample(lat, lon)
    # Stay away from the painted boundary and periodic outer map edge.
    np.testing.assert_array_equal(source.sample(0, 90), conditioning.RandomElevation(0).sample(0, 90))
    np.testing.assert_array_equal(source.sample(0, -90), np.float32(6000))
    pixels[:, 8:, :3] = 0
    Image.fromarray(pixels).save(path)
    other = conditioning.make_source('mixed', 0, path, -6000, 6000, map_weight=1)
    np.testing.assert_array_equal(actual, other.sample(lat, lon))
    assert not np.array_equal(actual, random)


def test_coarse_conditions_mix_before_encoding_and_preserve_cell_centres(tmp_path):
    path = tmp_path / 'map.png'
    Image.new('L', (32, 16), 180).save(path)
    grid = Grid(64, 16, 8, factor=4)
    args = (16, 4, grid.coarse_halo)
    map_values = coarse_condition_cells(conditioning.make_source('map', 0, path), *args)
    random_values = coarse_condition_cells(conditioning.make_source('random', 0), *args)
    mixed = coarse_condition_cells(conditioning.make_source('mixed', 0, path, map_weight=.75), *args)
    np.testing.assert_allclose(mixed, .75 * map_values + .25 * random_values, atol=.002)


def test_cache_identity_pins_source_and_mix_but_accepts_legacy_map_runs(tmp_path):
    path = tmp_path / 'map.png'
    Image.new('L', (16, 8), 128).save(path)
    map_settings = conditioning.source_settings('map', path)
    legacy = {'dem_sha256': map_settings['dem_sha256']}
    assert conditioning.settings_differences(legacy, map_settings) == {}
    mixed = conditioning.source_settings('mixed', path, .5)
    assert 'input_source' in conditioning.settings_differences(legacy, mixed)
    assert 'input_source' in conditioning.settings_differences(mixed, map_settings)
    assert 'map_weight' in conditioning.settings_differences(mixed, conditioning.source_settings('mixed', path, .6))
    random = conditioning.source_settings('random')
    assert random['dem_sha256'] is None
    assert random['random_source_sha256'] == mixed['random_source_sha256']


def test_revised_random_guide_has_structure_near_training_input_spacing():
    from toy_meso.conditioning_v1 import RandomElevation as SmoothElevation
    # Compare adjacent ~23 km samples across a fixed global spatial population.
    lat = np.linspace(-70, 70, 50)[:, None]
    lon = np.linspace(-179, 179, 100)[None, :]
    roughness = []
    for source in (SmoothElevation(4100897701), conditioning.RandomElevation(4100897701)):
        a, b = source.sample(lat, lon), source.sample(lat + .207, lon)
        land = (a > 0) & (b > 0)
        roughness.append(np.sqrt(np.mean((a[land] - b[land]) ** 2)))
    assert roughness[1] > 2 * roughness[0]
