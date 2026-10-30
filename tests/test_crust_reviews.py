"""The reviews' label-free continent (twenty-second pass, agreed 25 Oct): total thickness >= 20 km."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import crust_review as cr

ROOT = Path(__file__).resolve().parents[1]
C1 = ROOT/'data'/'crust1'


@pytest.mark.skipif(not (C1/'crsthk.xyz').exists(), reason='CRUST1.0 not present')
def test_label_free_rule_reproduces_crust1_types_on_earth():
    e = json.loads((C1/'crust1-continental-summary.json').read_text(encoding='utf-8'))
    m = cr.earth_label_free()
    assert abs(m['area']-e['continental_area_fraction']) < .005
    assert abs(m['thin_lt30']-e['continental_share_thinner_than_30km']) < .005
    assert abs(m['thick_gt45']-e['continental_share_thicker_than_45km']) < .005
    assert abs(m['median_km']-e['continental_total_thickness_km_percentiles']['50']) <= .5


def test_column_thickness_counts_sediment_on_both_sides():
    z = {'continental': np.array([1, 1, 0, 0], np.uint8),
         'crust_thickness_m': np.array([35e3, 18e3, 7e3, 15e3]), 'sediment_m': np.array([2e3, 3e3, 1e3, 6e3])}
    with_layer = cr.column_thickness_m(z, {'continental_sediment': True})
    assert np.allclose(with_layer, [37e3, 21e3, 8e3, 21e3])
    # before the sediment layer, deposits on continents were booked as crust: the sediment field there is not a layer
    assert np.allclose(cr.column_thickness_m(z, {'continental_sediment': False}), [35e3, 18e3, 8e3, 21e3])
    # a thinned margin (flag continental, 18 km) is ocean to the rule; a thick sediment pile on ocean crust is continent
    assert list(cr.label_free_continent(with_layer)) == [True, True, False, True]
