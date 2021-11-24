import operator as op

import pytest
import mock
import numpy as np

import allensdk.internal.mouse_connectivity.tissuecyte_stitching.stitcher as stitcher


def test_initialize_image():

    image = stitcher.initialize_image({'row': 40, 'column': 12}, 2, np.float32, 'F')

    assert( np.allclose( image.shape, [40, 12, 2] ) ) 
    assert( np.sum(image) == 0 )
    assert( image.dtype == np.float32 )
    assert( image.flags.f_contiguous )


def test_initialize_images():

    a, b, = stitcher.initialize_images({'row': 40, 'column': 12}, 1)

    assert(a.dtype == np.uint16)
    assert(b.dtype == np.int8)
    assert(len(a.shape) == 3)


def test_make_blended_tile():

    tile = np.arange(25, dtype=float).reshape(5, 5)
    current_region = np.ones((5, 5)) * 30
    blend = np.zeros((5, 5))
    blend[-1, :] = 1
    blend[-2, :] = 0.5

    exp = tile.copy()
    exp[-1, :] = 30
    exp[-2, :] = (np.arange(15, 20, dtype=float) + 30) / 2
  
    obt = stitcher.make_blended_tile(blend, tile, current_region)
    assert(np.allclose(exp, obt))


@pytest.mark.parametrize('lg,axis,point', [(op.lt, 0, 0), (op.gt, 0, 8), (op.lt, 1, 1), (op.gt, 1, 7)])
def test_get_indicator_bound_point(lg, axis, point):

    indicator = np.zeros((10, 10))
    indicator[9:, :] = 1
    indicator[:, 8:] = 1
    indicator[0, :] = 1
    indicator[:, :2] = 1
    indicator[7:, 7:] = 0

    obt = stitcher.get_indicator_bound_point(indicator, lg, axis)
    
    assert( obt == point )


def test_blend_component_from_point():

    mesh = np.tile(np.arange(20), (10, 1))
    point = 15 # indexed in the diff: 17 -> only last r/c
    lg = op.gt

    exp = np.zeros((10, 20))
    exp[:, :18] = 0
    exp[:, -3] = 1.0 / 3.0
    exp[:, -2] = 2.0 / 3.0
    exp[:, -1] = 1

    obt = stitcher.blend_component_from_point(point, mesh, lg)
    assert( np.allclose( obt, exp ) )


def test_blend_component_from_point_divzero():

    mesh = np.zeros((20, 20))
    point = 0

    lg = op.gt
    obt = stitcher.blend_component_from_point(point, mesh, lg)
    
    assert( np.allclose(obt, mesh) )


def test_get_blend_component_nopoint():

    with mock.patch('allensdk.internal.mouse_connectivity.tissuecyte_stitching.stitcher.get_indicator_bound_point', 
                    new=lambda *a, **k: None):

        assert( len(stitcher.get_blend_component(1, 2, 3, 4)) == 0 )


def test_get_blend_component_actual():

    indicator = np.zeros((20, 10))
    indicator[16:, :] = 1

    lg = op.gt
    axis = 0
    meshes = np.meshgrid(np.arange(20), np.arange(10), indexing='ij')

    exp = np.zeros_like(indicator)
    exp[17, :] = 1.0 / 3.0 
    exp[18, :] = 2.0 / 3.0
    exp[19, :] = 1.0
    
    obt = stitcher.get_blend_component(indicator, lg, axis, meshes)
    assert( np.allclose( obt, exp ) )


def test_get_overall_blend():

    meshes = np.meshgrid(np.arange(20), np.arange(20), indexing='ij')

    indicator = np.zeros((20, 20))
    indicator[16:, :] = 1
    indicator[:, 17:] = 1

    exp = np.zeros_like(indicator)
    exp[17, :] = 1.0 / 3.0
    exp[:, 18] = 1.0 / 2.0
    exp[18, :] = 2.0 / 3.0
    exp[:, -1] = 1
    exp[-1, :] = 1
    
    obt = stitcher.get_overall_blend(indicator, meshes)
    assert( np.allclose(obt, exp) )


def test_get_blend():

    indicator = np.zeros((20, 10))
    indicator[16:, :] = 1
    indicator[:, 7:] = 1

    stup = (20, 10)
    cb = np.sqrt

    exp = np.zeros_like(indicator)
    exp[17, :] = 1.0 / 3.0
    exp[:, 8] = 1.0 / 2.0
    exp[18, :] = 2.0 / 3.0
    exp[:, -1] = 1
    exp[-1, :] = 1
    exp = np.sqrt(exp)

    obt = stitcher.get_blend(indicator, stup, cb)
    assert( np.allclose( obt, exp ) )
