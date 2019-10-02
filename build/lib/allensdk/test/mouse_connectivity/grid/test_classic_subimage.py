import pytest
import mock

import numpy as np

from allensdk.mouse_connectivity.grid.subimage import ClassicSubImage


#==============================================================================
#==============================================================================


@pytest.fixture(scope='function') 
def classic_params():
    return {'reduce_level': 4, 
            'in_dims': np.array([30000, 40000]), 
            'in_spacing': np.array([0.35, 0.35]), 
            'coarse_spacing': np.array([16.8, 16.8]), 
            'segmentation_paths': {'segmentation': '/path/to_segmentation'}, 
            'intensity_paths': {'green': {'path': '/path/to/intensity', 'channel': 1}}, 
            'polygon_info': {'missing_tile': [[(8000, 8000), (16000, 8000), 
                                               (16000, 16000), (8000, 16000)]], 
                             'no_signal': [(4000, 4000), (8000, 4000), 
                                           (8000, 8000), (4000, 8000)]}}


@pytest.fixture(scope='function')
def missing_tile():
    image = np.zeros([1875, 2500], dtype=np.uint8)
    image[500:1000, 500:1000] = 1
    return image


@pytest.fixture(scope='function')
def no_signal():
    image = np.zeros([1875, 2500], dtype=np.uint8)
    image[250:500, 250:500] = 1
    return image


@pytest.fixture(scope='function')
def segmentation_image():

    image = np.zeros([1875, 2500], dtype=np.uint8)
    image[:1000, :] += 1
    image[:, :1000] += 128
    image[:20, :20] = 64

    return image


@pytest.fixture(scope='function')
def projection():

    image = np.zeros([1875, 2500], dtype=np.uint8)

    image[:, :1000] = 1
    image[:20, :20] = 0
    image[500:1000, 500:1000] = 0
    image[250:500, 250:500] = 0

    return image


@pytest.fixture(scope='function')
def injection():

    image = np.zeros([1875, 2500], dtype=np.uint8)

    image[:1000, :] = 1
    image[:20, :20] = 0
    image[500:1000, 500:1000] = 0

    return image


#==============================================================================
#==============================================================================


def test_init_classic(classic_params):

    si = ClassicSubImage(**classic_params)

    assert( hasattr(si, 'intensity_paths') )
    assert( hasattr(si, 'segmentation_paths') )
    assert( hasattr(si, 'polygon_info') )


def test_process_segmentation(classic_params, segmentation_image, 
                              missing_tile, no_signal, projection, injection):

    si = ClassicSubImage(**classic_params)
    si.images['segmentation'] = segmentation_image
    si.images['missing_tile'] = missing_tile
    si.images['no_signal'] = no_signal

    si.process_segmentation()

    assert(np.allclose( projection, si.images['projection'] ))
    assert(np.allclose( injection, si.images['injection'] ))
    assert( 'segmentation' not in si.images )
  

def test_compute_intensity(classic_params):

    pr = np.zeros([1875, 2500])
    pr[:600, :] = 1

    ij = np.zeros([1875, 2500])
    ij[:, :600] = 1

    si = ClassicSubImage(**classic_params)
    si.images['green'] = np.ones([1875, 2500]) * 2
    si.images['projection'] = pr
    si.images['injection'] = ij

    si.compute_intensity()

    spi_expected = np.ones([625, 834]) * 144 * 2
    spi_expected[:, -1] = 48 * 2

    ispi_expected = np.zeros_like(spi_expected)
    ispi_expected[:, :200] = spi_expected[:, :200]

    sppi_expected = np.zeros_like(spi_expected)
    sppi_expected[:200, :] = spi_expected[:200, :]

    isppi_expected = np.zeros_like(spi_expected)
    isppi_expected[:200, :200] = spi_expected[:200, :200]
    
    assert(np.allclose( spi_expected * 2, si.accumulators['sum_pixel_intensities'] ))
    assert(np.allclose( ispi_expected * 2, si.accumulators['injection_sum_pixel_intensities'] ))
    assert(np.allclose( sppi_expected * 2, si.accumulators['sum_projecting_pixel_intensities'] ))
    assert(np.allclose( isppi_expected * 2, si.accumulators['injectionsum_projecting_pixel_intensities'] ))
    assert( 'intensity' not in si.images )


def test_compute_injection(classic_params):

    pr = np.zeros([1875, 2500])
    pr[:600, :] = 1

    ij = np.zeros([1875, 2500])
    ij[:, :600] = 1

    si = ClassicSubImage(**classic_params)
    si.images['projection'] = pr
    si.images['injection'] = ij

    si.compute_injection()

    isp_expected = np.zeros([625, 834])
    isp_expected[:, :200] = 144

    ispp_expected = np.zeros([625, 834])
    ispp_expected[:200, :200] = 144

    assert(np.allclose( isp_expected * 2, si.accumulators['injection_sum_pixels'] ))
    assert(np.allclose( ispp_expected * 2, si.accumulators['injection_sum_projecting_pixels'] ))
    assert( 'injection' not in si.images )


def test_compute_projection(classic_params):

    pr = np.zeros([1875, 2500])
    pr[:600, :] = 1

    si = ClassicSubImage(**classic_params)
    si.images['projection'] = pr

    si.compute_projection()

    spp_expected = np.zeros([625, 834])
    spp_expected[:200, :] = 144
    spp_expected[:200, -1] = 48
    
    assert(np.allclose( spp_expected * 2, si.accumulators['sum_projecting_pixels'] ))
    assert( 'projection' not in si.images )


def test_compute_sum_pixels_aav(classic_params):

    mt = np.zeros([1875, 2500])
    mt[:300, :300] = 1
    
    aav = np.zeros([1875, 2500])
    aav[300:600, :] = 1

    si = ClassicSubImage(**classic_params)
    si.images['missing_tile'] = mt
    si.images['aav_exclusion'] = aav

    si.compute_sum_pixels()

    sp_expected = np.zeros([625, 834]) + 144
    sp_expected[:100, :100] = 0
#    sp_expected[100, :101] = 48
#    sp_expected[:101, 100] = 48
    sp_expected[:, -1] = 48
 
    aesp_expected = np.zeros([625, 834])
    aesp_expected[100:200, :] = 144
    aesp_expected[100:200, -1] = 48

    assert(np.allclose( aesp_expected * 2, si.accumulators['aav_exclusion_sum_pixels'] ))
    assert(np.allclose( sp_expected * 2, si.accumulators['sum_pixels'] ))   

