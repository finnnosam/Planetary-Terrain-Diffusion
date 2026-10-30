import numpy as np
import pytest

from planet_diffusion import terrain_metrics as metrics


def terrain(wavelength, size=256, spacing=8000):
    x = np.arange(size)*spacing/1000
    return np.broadcast_to(1500+1000*np.sin(2*np.pi*x/wavelength),(size,size)).copy()


def test_horizontal_stretch_changes_scale_without_height_change():
    a = metrics.spectrum(terrain(96),8000,edges_km=(32,64,128,256,512))
    b = metrics.spectrum(terrain(192),8000,edges_km=(32,64,128,256,512))
    assert np.argmax([r['variance_m2'] for r in a['bands']]) == 1
    assert np.argmax([r['variance_m2'] for r in b['bands']]) == 2
    assert a['bands'][1]['fraction_of_resolved_band_variance'] > .95
    assert b['bands'][2]['fraction_of_resolved_band_variance'] > .95


def test_height_scaling_does_not_change_horizontal_spectrum():
    a = metrics.spectrum(terrain(96),8000)
    b = metrics.spectrum(terrain(96)*2,8000)
    for original,scaled in zip(a['bands'],b['bands']):
        np.testing.assert_allclose(scaled['variance_m2'],4*original['variance_m2'],rtol=1e-10)
        np.testing.assert_allclose(scaled['fraction_of_resolved_band_variance'],original['fraction_of_resolved_band_variance'],rtol=1e-10)
    for original,scaled in zip(a['lags'],b['lags']):
        np.testing.assert_allclose(scaled['correlation'],original['correlation'],rtol=1e-10)
    np.testing.assert_allclose(a['resolved_band_fraction_of_windowed_power'],
                               b['resolved_band_fraction_of_windowed_power'],rtol=1e-10)


def test_same_physical_terrain_at_different_grid_resolutions():
    a = metrics.spectrum(terrain(96,256,8000),8000,edges_km=(32,64,128,256,512))
    b = metrics.spectrum(terrain(96,512,4000),4000,edges_km=(32,64,128,256,512))
    np.testing.assert_allclose([r['fraction_of_resolved_band_variance'] for r in a['bands']],
                               [r['fraction_of_resolved_band_variance'] for r in b['bands']],atol=.005)


def test_high_flat_plateau_is_not_rugged():
    a = np.full((64,128),3000.,np.float32)
    rough = metrics.ruggedness(a)
    assert rough.max() < .001
    assert metrics.belt_geometry(rough>600) == []


def test_longitude_seam_does_not_split_a_belt():
    mask = np.zeros((8,16),bool)
    mask[3:5,:2] = True
    mask[3:5,-2:] = True
    labels = metrics.components(mask)
    assert labels[3,0] == labels[3,-1] != 0
    assert len(np.unique(labels[mask])) == 1
    mask[3:5,7:9] = True
    labels = metrics.components(mask)
    assert len(np.unique(labels[mask])) == 2


def test_width_and_radius_scaling_are_measured_separately():
    mask = np.zeros((360,720),bool)
    mask[179:181,320:400] = True
    a = metrics.belt_geometry(mask)[0]
    mask[178:182,320:400] = True
    b = metrics.belt_geometry(mask)[0]
    np.testing.assert_allclose(b['effective_width_km']/a['effective_width_km'],2,rtol=.01)
    np.testing.assert_allclose(b['length_km']/a['length_km'],1,rtol=.01)
    c = metrics.belt_geometry(mask,radius=2*6371000)[0]
    np.testing.assert_allclose(c['length_km'],2*b['length_km'],rtol=1e-10)
    np.testing.assert_allclose(c['effective_width_km'],2*b['effective_width_km'],rtol=1e-10)
    np.testing.assert_allclose(c['area_km2'],4*b['area_km2'],rtol=1e-10)


def test_spectral_limits_do_not_claim_unresolved_wavelengths():
    a = metrics.spectrum(terrain(96,128,8000),8000,edges_km=(8,16,32,64,128,256,512))
    assert [r['wavelength_km'] for r in a['bands']] == [[32,64],[64,128],[128,256]]
    assert a['supported_wavelength_km'] == [32,256]


def test_local_projection_radial_distances():
    p = metrics.patch_directions(30,40,64,8000)
    centre = metrics.directions(*np.deg2rad([30,40]))
    angle = np.arccos(np.clip(p@centre,-1,1))
    xy = (np.arange(64)+.5-32)*8000
    np.testing.assert_allclose(angle*6371000,np.hypot(xy[:,None],xy[None,:]),atol=1e-5)


def test_clipped_belt_is_reported_separately():
    mask = np.zeros((180,360),bool)
    mask[10:50,175:185] = True
    assert metrics.belt_geometry(mask) == []
    rows = metrics.belt_geometry(mask,include_censored=True)
    assert len(rows) == 1
    assert rows[0]['clipped_by_latitude_boundary'] is True


@pytest.mark.parametrize('spacing',[0,-1,np.nan])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        metrics.spectrum(terrain(96),spacing)
