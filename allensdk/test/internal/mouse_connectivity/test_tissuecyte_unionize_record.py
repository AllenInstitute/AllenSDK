from __future__ import division

import numpy as np
import pytest
import mock
from six import iteritems
from six.moves import xrange

from allensdk.internal.mouse_connectivity.interval_unionize\
    .tissuecyte_unionize_record import TissuecyteBaseUnionize, \
    TissuecyteInjectionUnionize,TissuecyteProjectionUnionize
    
  
@pytest.fixture(scope='function')  
def data_arrays():

    fq = np.ones(100)
    fq[25:] = 0
    
    lq = np.ones(100)
    lq[:75] = 0

    top = np.ones(100)
    top[:70] = 0

    return {'injection_fraction': fq, 
            'aav_exclusion_fraction': top, 
            'projection_density': np.arange(100), 
            'projection_energy': np.arange(100) * 2, 
            'injection_density': np.multiply(np.arange(100), fq), 
            'injection_energy': np.multiply(np.arange(100) * 2, fq), 
            'sum_pixels': np.ones(100) * 900,
            'sum_pixel_intensities': np.ones(100),
            'injection_sum_pixel_intensities': np.ones(100)[:50]
            }
    
    
def test_base_init():

    tbu = TissuecyteBaseUnionize()
    for item in TissuecyteBaseUnionize.__slots__:
        assert( getattr(tbu, item) == 0 )
        

@pytest.mark.parametrize('anc_mvd', [0, 1])
def test_base_propagate(anc_mvd):
        
    an = TissuecyteBaseUnionize()
    an.sum_pixels = 12
    an.max_voxel_index = 100
    an.max_voxel_density = anc_mvd
    
    ch = TissuecyteBaseUnionize()
    ch.sum_pixels = 5
    ch.max_voxel_index = 50
    ch.max_voxel_density = 0.5
    
    ch.propagate(an)
    
    assert( an.sum_pixels == 17 )
    
    if an.max_voxel_density == 1:
        assert( an.max_voxel_index == 100 )
    else:
        assert( an.max_voxel_index == 50 )
        
     
@pytest.mark.parametrize('spp', [0, 1])   
def test_base_set_max_voxel(spp):

    darr = np.arange(25) / 24 
    darr[15:] = 0
    low = 12

    tbu = TissuecyteBaseUnionize()
    tbu.sum_projection_pixels = spp
    tbu.set_max_voxel(darr, low)
    
    if spp == 1:
        assert( tbu.max_voxel_index == 26 )
        assert( tbu.max_voxel_density == 14 / 24 )
    else:
        assert( tbu.max_voxel_index == 0 )
        assert( tbu.max_voxel_density == 0 )
        
        
def test_base_slice_arrays():

    arrays = {ii: np.arange(10) + ii for ii in xrange(20)}
    low = 5
    high = 8
    
    tbu = TissuecyteBaseUnionize()
    sl = tbu.slice_arrays(low, high, arrays)
    
    for k, v in iteritems(sl):
        assert( len(v) == 3 )
        assert( v.sum() == k * 3 + 18 )
        
        
@pytest.mark.parametrize('sum_pixels,sum_projection_pixels', [(0, 2), (0, 2)])
def test_base_output(sum_pixels, sum_projection_pixels):

    tbu = TissuecyteBaseUnionize()
    
    tbu.sum_pixels = sum_pixels
    tbu.direct_sum_projection_pixels = sum_projection_pixels / 2
    tbu.sum_projection_pixels = sum_projection_pixels
    tbu.sum_projection_pixel_intensity = 100
    
    tbu.max_voxel_index = 999
    tbu.max_voxel_density = 1
    
    out = tbu.output(10, 900, (10, 10, 10), np.arange(1000))
    
    assert( out['volume'] == sum_pixels * 900 )
    assert( out['direct_projection_volume'] == sum_projection_pixels * 450 )
    assert( out['projection_volume'] == sum_projection_pixels * 900 )
    
    if sum_pixels > 0:
        assert( out['projection_density'] == sum_projection_pixels / sum_pixels )
    else:
        assert( out['projection_density'] == 0 )
        
    if sum_pixels > 0:
        assert( out['projection_energy'] == 100 / sum_pixels )
    else:
        assert( out['projection_energy'] == 0 )
        
    if sum_projection_pixels > 0:
        assert( out['projection_intensity'] == 100 / sum_projection_pixels )
    else:
        assert( out['projection_intensity'] == 0 )
        
    assert( out['max_voxel_x'] == 90 )
    assert( out['max_voxel_y'] == 90 )
    assert( out['max_voxel_z'] == 90 )
    
    
def test_injection_calculate(data_arrays):

    tiu = TissuecyteInjectionUnionize()
    
    tiu.calculate(20, 80, data_arrays)
    
    assert( tiu.sum_pixels == 4500 )
    assert( tiu.sum_projection_pixels == 900 * np.arange(20, 25).sum() )
    assert( tiu.sum_projection_pixel_intensity == 1800 * np.arange(20, 25).sum() )
    
    assert( tiu.max_voxel_index == 24 )
    assert( tiu.max_voxel_density == 24 )
    
    
def test_projection_calculate(data_arrays):

    tiu = mock.MagicMock()
    tiu.sum_pixels = 1
    tiu.sum_projection_pixels = 2
    tiu.sum_projection_pixel_intensity = 3
    
    tpu = TissuecyteProjectionUnionize()
    tpu.calculate(20, 80, data_arrays, tiu)
    
    assert( tpu.sum_pixels == 900 * 50 - 1 )
    assert( tpu.sum_projection_pixels == 900 * np.arange(20, 70).sum() - 2 )
    assert( tpu.sum_projection_pixel_intensity == 1800 * np.arange(20, 70).sum() - 3  )
    
    assert( tpu.max_voxel_index == 69 )
    assert( tpu.max_voxel_density == 69 )

