import pytest
import mock
from six.moves import range

import numpy as np
import SimpleITK as sitk

from allensdk.mouse_connectivity.grid.image_series_gridder import ImageSeriesGridder


def small_gridder():

    in_dims = [12353, 16471, 140]
    in_spacing = [0.85, 0.85, 100.0]
    out_dims = [1320, 800, 1140]
    out_spacing = [10.0, 10.0, 10.0]
    reduce_level = 0
    
    subimages = [{'index': 12, 
                  'segmentation_path': '/path/to/projection_12.jp2', 
                  'intensity_path': '/path/to/image_12.jp2', 
                  'polygon_info': {'missing_tile': [], 
                                   'no_signal': [], 
                                   'aav_exclusion': []}}]
    
    subimage_kwargs = {'cls': dict, 'channel': 1}
    
    nprocesses = 8
    
    affine_params = [2, 0 , 0, 0, 2, 0, 0, 0, 2, 1, 1, 1]
    dfmfld_path = '/path/to/deformation_field_header.mhd'


    return ImageSeriesGridder(in_dims, in_spacing, out_dims, out_spacing, 
                              reduce_level, subimages, subimage_kwargs, 
                              nprocesses, affine_params, dfmfld_path)
                              

def large_gridder():
    
    in_dims = [30000, 40000, 140]
    in_spacing = [0.35, 0.35, 100.0]
    out_dims = [1320, 800, 1140]
    out_spacing = [10.0, 10.0, 10.0]
    reduce_level = 1
    
    subimages = [{'index': 12, 
                  'segmentation_path': '/path/to/projection_12.jp2', 
                  'intensity_path': '/path/to/image_12.jp2', 
                  'polygon_info': {'missing_tile': [], 
                                   'no_signal': [], 
                                   'aav_exclusion': []}}]
    
    subimage_kwargs = {'cls': dict, 'channel': 1}
    
    nprocesses = 8
    
    affine_params = [2, 0 , 0, 0, 2, 0, 0, 0, 2, 1, 1, 1]
    dfmfld_path = '/path/to/deformation_field_header.mhd'


    return ImageSeriesGridder(in_dims, in_spacing, out_dims, out_spacing, 
                              reduce_level, subimages, subimage_kwargs, 
                              nprocesses, affine_params, dfmfld_path)
                              
                              
@pytest.mark.parametrize('gridder_fn,cgd,cgs,cgr', [(small_gridder, [951, 1267, 140], [11.05, 11.05, 100], 6), 
                                                    (large_gridder, [1000, 1334, 140], [10.5, 10.5, 100], 7)])
def test_set_coarse_grid_parameters(gridder_fn, cgd, cgs, cgr):

    gridder = gridder_fn()
    gridder.set_coarse_grid_parameters()
    
    assert(np.allclose( gridder.coarse_dims, cgd ))
    assert(np.allclose( gridder.coarse_spacing, cgs ))
    assert(np.allclose( gridder.coarse_grid_radius, cgr ))


@pytest.mark.parametrize('gridder_fn,reduce_level', [(small_gridder, 0), (large_gridder, 1)])
def test_setup_subimages(gridder_fn, reduce_level):

    gridder = gridder_fn()
    gridder.setup_subimages()
    
    for s in gridder.subimages:
        assert( s['reduce_level'] == reduce_level )
    gridder = gridder_fn()
    

def test_initialize_coarse_volume():
    
    key = 'amethystine'
    size = [1, 2, 3]
    spacing = [4, 5, 6]
    
    gridder = small_gridder()
    gridder.coarse_dims = size
    gridder.coarse_spacing = spacing
    
    gridder.initialize_coarse_volume(key, sitk.sitkFloat32)
    
    assert(np.allclose( gridder.volumes[key].GetSize(), size ))
    assert(np.allclose( gridder.volumes[key].GetSpacing(), spacing ))
    
    
def test_paste_slice():
    
    key = 'halmahera'
    slice_array = np.eye(1000)
    index = 12
    
    volume = sitk.Image(1000, 1000, 140, sitk.sitkFloat32)
    volume.SetSpacing([1, 1, 100])
    
    gridder = small_gridder()
    gridder.coarse_spacing = [1, 1, 100]
    gridder.volumes[key] = volume
    
    gridder.paste_slice(key, index, slice_array)
    
    obt = sitk.GetArrayFromImage(gridder.volumes[key])
    assert(np.allclose( obt[12, :, :], slice_array ))
    
    
def test_paste_subimage():
    
    index = 1
    output = {'a': 1, 'b': 2}
    
    gridder = small_gridder()
    gridder.paste_slice = mock.MagicMock()
    
    gridder.paste_subimage(index, output)
    
    assert( len(gridder.paste_slice.mock_calls) == 2 )
    assert( output['a'] is None and output['b'] is None )
    
    
def test_build_coarse_grids():
    # mp is hard to test

    class Dummy(object):
        
        def __init__(self, *a, **k):
            pass
    
        def imap_unordered(*a, **k):
            for ii in range(20):
                yield ii, ii

    with mock.patch('multiprocessing.Pool', new=Dummy) as p:

        gridder = small_gridder()
        gridder.paste_subimage = mock.MagicMock()
        
        gridder.build_coarse_grids()
        
        for ii in range(20):
            assert( mock.call(ii, ii) in gridder.paste_subimage.mock_calls )
        
    
def test_resample_volume():
    
    def make_dfield(*a , **k):
        return
        
    def make_transform(*a, **k):
        return sitk.TranslationTransform(3, [2, 2, 2])
        
    key = 'green_tree'
        
    volume = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    volume.SetSpacing([1, 1, 1])
    volume += 1
        
    with mock.patch('SimpleITK.ReadImage', new=make_dfield) as p:
        with mock.patch(
            'allensdk.mouse_connectivity.grid.utilities.image_utilities.build_composite_transform', 
            new=make_transform
        ) as q:
                        
            gridder = small_gridder()
            gridder.out_dims = [10, 10, 10]
            gridder.out_spacing = [1, 1, 1]
            
            gridder.volumes[key] = volume 
            gridder.resample_volume(key)
            
            arr = sitk.GetArrayFromImage(gridder.volumes[key])
            assert( arr.sum() == 8**3 )
            
