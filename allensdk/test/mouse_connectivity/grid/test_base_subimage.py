import sys

import pytest
import mock

import numpy as np

sys.modules['jpeg_twok'] = mock.Mock()
from allensdk.mouse_connectivity.grid.subimage.base_subimage import SubImage, \
    SegmentationSubImage, IntensitySubImage, PolygonSubImage


#==============================================================================
#==============================================================================


@pytest.fixture(scope='function') 
def base_params():
    return {'reduce_level': 4, 
            'in_dims': np.array([30000, 40000]), 
            'in_spacing': np.array([0.35, 0.35]), 
            'coarse_spacing': np.array([16.8, 16.8])}


@pytest.fixture(scope='function') 
def segmentation_params():
    return {'reduce_level': 4, 
            'in_dims': np.array([30000, 40000]), 
            'in_spacing': np.array([0.35, 0.35]), 
            'coarse_spacing': np.array([16.8, 16.8]), 
            'segmentation_paths': {'name_one': 'path_one', 
                                   'name_two': 'path_two'}}


@pytest.fixture(scope='function') 
def intensity_params():
    return {'reduce_level': 4, 
            'in_dims': np.array([30000, 40000]), 
            'in_spacing': np.array([0.35, 0.35]), 
            'coarse_spacing': np.array([16.8, 16.8]), 
            'intensity_paths': {'name_one': {'path': 'path_one', 'channel': 2}}}


@pytest.fixture(scope='function') 
def polygon_params():
    return {'reduce_level': 4, 
            'in_dims': np.array([30000, 40000]), 
            'in_spacing': np.array([0.35, 0.35]), 
            'coarse_spacing': np.array([16.8, 16.8]), 
            'polygon_info': {'hello_am_square': [[(8000, 8000), (16000, 8000), 
                                                 (16000, 16000), (8000, 16000)]]}}


#==============================================================================
#==============================================================================

            
def test_init_base(base_params):

    si = SubImage(**base_params)
    
    assert(np.allclose( si.coarse_dims, [625, 834] ))


def test_binarize(base_params):

    si = SubImage(**base_params)

    si.images['fish'] = np.arange(25).reshape([5, 5])
    si.binarize('fish')

    expected = np.ones([5, 5])
    expected[0, 0] = 0 

    assert( np.allclose(si.images['fish'], expected) )


@pytest.mark.parametrize('positive', [(True), (False)])
def test_apply_mask(base_params, positive):

    si = SubImage(**base_params)

    si.images['submarine'] = np.arange(25).reshape([5, 5])
    si.images['aquaman'] = np.eye(5).astype(np.uint8)
    si.images['aquaman'][3, 3] = 0
    
    if positive:
      exp = [0, 6, 12, 0, 24]
    else:
      exp = [0, 0, 0, 18, 0]

    si.apply_mask('submarine', 'aquaman', positive)
    assert( np.allclose( np.diag(si.images['submarine']), exp ) )


def test_make_pixel_counter(base_params):

    si = SubImage(**base_params)
    reducer = si.make_pixel_counter()

    img = np.zeros([10000, 13334])
    img[::3, :] = 1

    reduced = reducer(img)

    exp = np.ones([625, 834]) * 48 * 2
    exp[:, -1] = 16 * 2

    assert(np.allclose(exp, reduced))    
    

#==============================================================================
#==============================================================================
    

def test_init_segmentation(segmentation_params):
    si = SegmentationSubImage(**segmentation_params)
    assert( si.segmentation_paths['name_one'] == 'path_one' )


def test_extract_signal_from_segmentation(segmentation_params):
  
    si = SegmentationSubImage(**segmentation_params)

    segmentation_name = 'fish'
    signal_name = 'fish_signal'

    si.images[segmentation_name] = np.arange(256)
    si.extract_signal_from_segmentation(segmentation_name, signal_name)
    
    exp = np.array([0] * 128 + [1] * 128)
    assert(np.allclose( exp, si.images[signal_name] ))
    

def test_extract_injection_from_segmentation(segmentation_params):

    si = SegmentationSubImage(**segmentation_params)


    segmentation_name = 'fish'
    injection_name = 'fish_injection'

    si.images[segmentation_name] = np.arange(256)
    si.extract_injection_from_segmentation(segmentation_name, injection_name)

    exp = np.arange(256)
    exp[exp % 32 == 0] = 0
    exp[exp > 0] = 1

    assert(np.allclose( exp, si.images[injection_name] ))



def test_read_segmentation_image(segmentation_params):

    si = SegmentationSubImage(**segmentation_params)
    arr = np.zeros((32, 32))
    arr[:16, :] = 1

    exp = np.array([[1, 1], [0, 0]])

    with mock.patch('allensdk.mouse_connectivity.grid.utilities.image_utilities.read_segmentation_image', return_value=arr) as p:
        si.read_segmentation_image('name_one')
        p.assert_called_once_with('path_one')

        assert(np.allclose( exp, si.images['name_one'] ))

#==============================================================================
#==============================================================================


def test_init_intensity(intensity_params):

    si = IntensitySubImage(**intensity_params)

    assert(si.intensity_paths['name_one']['path'] == 'path_one')
    assert(si.intensity_paths['name_one']['channel'] == 2)


def test_get_intensity(intensity_params):

    arr = np.eye(1000)
    arr[999, 0] = 1

    si = IntensitySubImage(**intensity_params)

    with mock.patch(
        'allensdk.mouse_connectivity.grid.subimage.base_subimage.IntensitySubImage.required_intensities', 
        new_callable=mock.PropertyMock
    ) as a:
        a.return_value = ['name_one']

        with mock.patch('allensdk.mouse_connectivity.grid.utilities.image_utilities.read_intensity_image', return_value=arr) as p:
            si.get_intensity()

            p.assert_called_once_with('path_one', 4, 2)
            assert(np.allclose( si.images['name_one'], arr ))


#==============================================================================
#==============================================================================


def test_init_polygon(polygon_params):

    si = PolygonSubImage(**polygon_params)

    assert(np.allclose( si.polygon_info['hello_am_square'], 
                        np.array([[(8000, 8000), (16000, 8000), (16000, 16000), (8000, 16000)]]) ))


def test_get_polygons(polygon_params):

    si = PolygonSubImage(**polygon_params)

    arr = np.zeros([1875, 2500])
    arr[500:1000, 500:1000] = 1

    with mock.patch(
        'allensdk.mouse_connectivity.grid.subimage.base_subimage.PolygonSubImage.required_polys', 
        new_callable=mock.PropertyMock
    ) as a:
        a.return_value = ['hello_am_square']

        si.get_polygons()
        assert(np.allclose( si.images['hello_am_square'], arr ))
    

#==============================================================================
#==============================================================================
    
