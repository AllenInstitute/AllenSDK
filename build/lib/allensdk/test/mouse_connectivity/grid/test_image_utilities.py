import pytest
import mock

import SimpleITK as sitk
import numpy as np

from allensdk.mouse_connectivity.grid.utilities import image_utilities as iu

@pytest.fixture(scope='function')
def dfmfld():

    disp = sitk.Image(10, 10, 10, sitk.sitkVectorFloat64)
    disp.SetSpacing([1, 1, 1])
    
    disp += 2
    
    return disp
    
    
@pytest.fixture(scope='function')
def aff_params():
    return [2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 1, 1]
    

def test_set_image_spacing():
    
    im = sitk.Image(5, 5, 5, sitk.sitkFloat32)
    
    iu.set_image_spacing(im, [1, 2, 3])
    
    assert( np.allclose(im.GetSpacing(), [1, 2, 3]) )
    assert( np.allclose(im.GetOrigin(), [0.5, 1, 1.5]) )
    
    
def test_new_image_3d():
    
    im = iu.new_image([300, 200, 100], [1, 2, 3], sitk.sitkFloat32)
    
    assert( np.allclose(im.GetSize(), [300, 200, 100]) )
    assert( np.allclose(im.GetSpacing(), [1, 2, 3]) ) 
    assert( np.allclose(im.GetOrigin(), [0.5, 1, 1.5]) )
    

def test_new_image_2d():
    
    im = iu.new_image([300, 200], [1, 2], sitk.sitkFloat32)
    
    assert( np.allclose(im.GetSize(), [300, 200]) )
    assert( np.allclose(im.GetSpacing(), [1, 2]) ) 
    assert( np.allclose(im.GetOrigin(), [0.5, 1]) )
    

@pytest.mark.parametrize("np_type,sitk_type", [(np.float32, sitk.sitkFloat32)])    
def test_np_sitk_convert(np_type, sitk_type):
    
    arr = np.zeros((100, 100), dtype=np_type)
    
    sitk_obt = iu.np_sitk_convert(arr.dtype)
    np_obt = iu.sitk_np_convert(sitk_obt)
    
    assert( sitk_obt == sitk_type )
    assert( np_obt == np_type )
    
    
def test_compute_coarse_parameters():
    
    in_dims = [1000, 2000]
    in_spacing = [5, 5]
    out_spacing = [100, 100]
    reduce_level = 2
    
    cgd_exp = [50, 100]
    cgs_exp = [100, 100]
    cgr_exp = [2, 2]
    
    cgd_obt, cgs_obt, cgr_obt = iu.compute_coarse_parameters(in_dims, in_spacing, out_spacing, reduce_level)
    
    assert( np.allclose(cgd_obt, cgd_exp) )
    assert( np.allclose(cgs_obt, cgs_exp) )
    assert( np.allclose(cgr_obt, cgr_exp) )
    
    
def test_block_apply():
    
    row_blocks = [(ii, jj) for ii, jj in zip(range(0, 10, 2), range(2, 12, 2))]
    col_blocks = [(ii, jj) for ii, jj in zip(range(0, 10, 5), range(5, 15, 5))]
    blocks = [row_blocks, col_blocks]
    
    in_image = np.ones((10, 10))
    out_shape = [5, 2]
    dtype = np.float32
    
    out_image = iu.block_apply(in_image, out_shape, dtype, blocks, np.sum)
    
    assert( np.allclose(out_image, np.ones([5, 2]) * 10) )
    
    
def test_grid_image_blocks():

    in_shape = [10, 10]
    in_spacing = [5, 5]
    out_spacing = [20, 20]
    
    os_exp = [3, 3]
    b_exp = [[(0, 4), (4, 8), (8, 10)]] * 2
    
    blocks, out_shape = iu.grid_image_blocks(in_shape, in_spacing, out_spacing)
    
    assert( np.allclose(b_exp, blocks) )
    assert( np.allclose(os_exp, out_shape) )
    
    
def test_rasterize_polygons():

    shape = [10, 10]
    scale = [1, 1]
    points_list = [ [ (4, 4), (6, 4), (6, 6), (4, 6) ] ]
    
    exp = np.zeros((10, 10))
    exp[4:6, 4:6] = 1
    
    obt = iu.rasterize_polygons(shape, scale, points_list)
    assert( np.allclose(obt, exp) )
    
    
def test_resample_into_volume():

    vol = sitk.Image(20, 20, 10, sitk.sitkFloat32)
    vol.SetSpacing([10, 10, 10])
    
    im = sitk.Image(20, 20, sitk.sitkFloat32)
    im.SetSpacing([10, 10])
    im += 5
    
    obt = iu.resample_into_volume(im, None, 5, vol)
    
    arr = sitk.GetArrayFromImage(obt)
    
    assert( arr[5, :, :].sum() == 2000 )
    assert( arr.sum() == 2000 )
    
    
def test_build_affine_transform(aff_params):
    
    point = (1, 2, 3)
    
    tf = iu.build_affine_transform(aff_params)
    tp = tf.TransformPoint(point)
    
    assert( np.allclose(tp, [3, 5, 7]) )
    assert( np.allclose(tf.GetParameters(), aff_params) )
    
    
def test_build_composite_transform(dfmfld, aff_params):

    point = (5, 5, 5)
    exp = (15, 15, 15)
    
    trans = iu.build_composite_transform(dfmfld, aff_params)
    obt = trans.TransformPoint(point)
    
    assert( np.allclose(exp, obt) )
    
    
def test_resample_volume():
    
    volume = np.ones((10, 10, 10)).astype(np.float32)
    dims = [10, 10, 10]
    spacing = [1, 1, 1]
    
    transform = sitk.TranslationTransform(3, [2, 2, 2])
    
    obt = iu.resample_volume(sitk.GetImageFromArray(volume), dims, spacing, transform=transform, interpolator=sitk.sitkNearestNeighbor) # id
    obt_arr = sitk.GetArrayFromImage(obt)
    
    assert( obt_arr.sum() == 8**3 )
