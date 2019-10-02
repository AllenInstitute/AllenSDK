import pytest
import mock

import numpy as np

from allensdk.mouse_connectivity.grid.subimage import CavSubImage


#==============================================================================
#==============================================================================


@pytest.fixture(scope='function') 
def cav_params():
    return {'reduce_level': 4, 
            'in_dims': np.array([30000, 40000]), 
            'in_spacing': np.array([0.35, 0.35]), 
            'coarse_spacing': np.array([16.8, 16.8]), 
            'polygon_info': {'missing_tile': [[(8000, 8000), (16000, 8000), 
                                               (16000, 16000), (8000, 16000)]], 
                             'cav_tracer': [(4000, 4000), (8000, 4000), 
                                           (8000, 8000), (4000, 8000)]}}


def test_compute_coarse_planes(cav_params):

    mt = np.ones([1875, 2500])
    mt[:300, :] = 0

    ct = np.zeros([1875, 2500])
    ct[:, :300] = 1

    si = CavSubImage(**cav_params)
    si.images['cav_tracer'] = ct
    si.images['missing_tile'] = mt

    si.compute_coarse_planes()

    cav_tracer_expected = np.zeros([625, 834])
    cav_tracer_expected[:100, :100] = 144

    sum_pixels_expected = np.zeros([625, 834])
    sum_pixels_expected[:100, :] = 144
    sum_pixels_expected[:100, -1] = 48

    assert(np.allclose( sum_pixels_expected * 2, si.accumulators['sum_pixels'] ))
    assert(np.allclose( cav_tracer_expected * 2, si.accumulators['cav_tracer'] ))
