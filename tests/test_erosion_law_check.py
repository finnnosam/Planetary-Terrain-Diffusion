import numpy as np

from planet_layer.sphere_graph import CubeSphere
from scripts import erosion_law_check as e


def test_basin_mean_and_score():
    """A basin's law rate is the mean of the node field over its polygon; the bias rule reads log10(law / measured)."""
    g = CubeSphere(64)
    lon, lat = e.lonlat(g.xyz)
    rate = np.where(lon > 0, 100., 10.)                    # 100 m/Myr east of Greenwich, 10 west
    feats = []
    for k in range(25):
        x0 = 10.+k
        feats.append({'type': 'Feature', 'properties': {'EBE_MMKYR': 100., 'AREA': 1e6},
                      'geometry': {'type': 'Polygon', 'coordinates': [[[x0, 0], [x0+3, 0], [x0+3, 3], [x0, 3], [x0, 0]]]}})
    feats.append({'type': 'Feature', 'properties': {'EBE_MMKYR': 10., 'AREA': 1e6},
                  'geometry': {'type': 'Polygon', 'coordinates': [[[-40, 0], [-37, 0], [-37, 3], [-40, 3], [-40, 0]]]}})
    rows = e.compare(g, rate, {'type': 'FeatureCollection', 'features': feats})   # no temp files (Windows temp access)
    assert len(rows) == 26
    assert all(abs(r['law_m_myr']-r['measured_m_myr']) < 1e-9 for r in rows)
    s = e.score(rows)
    assert s['classes']['100-500']['scored'] and abs(s['classes']['100-500']['median_log10_ratio']) < 1e-9
    assert s['passes']
    s2 = e.score([{**r, 'law_m_myr': r['law_m_myr']*2} for r in rows])   # a factor 2 is outside +-0.2 in log10
    assert not s2['passes']
