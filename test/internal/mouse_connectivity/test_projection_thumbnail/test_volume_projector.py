import pytest
import numpy as np
import SimpleITK as sitk

from allensdk.internal.mouse_connectivity.projection_thumbnail.volume_projector import VolumeProjector


@pytest.fixture
def simple_volume():
    arr = np.arange(8 * 9 * 10, dtype=np.float64).reshape([8, 9, 10])
    return sitk.GetImageFromArray(arr) # swaps 0 <=> 2 axes


@pytest.fixture
def cube_volume():
    arr = np.arange(1000, dtype=np.float64).reshape([10, 10, 10])
    return sitk.GetImageFromArray(arr)


def test_init(simple_volume):
    vp = VolumeProjector(simple_volume)
    assert(vp.view_volume.GetPixel(3, 5, 3) == simple_volume.GetPixel(3, 5, 3))


def test_build_rotation_transform(simple_volume):

    vp = VolumeProjector(simple_volume)
    trans_obt = vp.build_rotation_transform(0, 2, np.pi / 2.0)

    exp = [0, 0, -1, 0, 1, 0, 1, 0, 0] # just hstacked rows
    assert(np.allclose(trans_obt.GetMatrix(), exp))


@pytest.mark.parametrize('angle,check', [(2*np.pi, [2, 3, 4]), (np.pi, [7, 3, 3])])
def test_rotate(simple_volume, angle, check):

    vp = VolumeProjector(simple_volume)
    obt = vp.rotate(0, 2, angle)

    assert(np.allclose(simple_volume.GetPixel(2, 3, 4), obt.GetPixel(*check)))
    

def test_extract():

    arr = np.eye(20)
    vp = VolumeProjector(arr)

    obt = vp.extract(np.sum)
    exp = 20

    assert(obt == exp)


@pytest.mark.parametrize('angle,exp', [(0.0, 0.0), (2 * np.pi, 0.0)])
def test_rotate_and_extract(angle, exp, cube_volume):

    cb = lambda x: x.GetPixel(0, 0, 0)
    vp = VolumeProjector(cube_volume)
    
    for obt in vp.rotate_and_extract([0], [2], [angle], cb):
        assert(np.allclose(obt, exp))


def test_fixed_factory(simple_volume):

    shape = [5, 6, 7]
    vp = VolumeProjector.fixed_factory(simple_volume, shape)
    
    shape_obt = vp.view_volume.GetSize()
    assert(np.allclose(shape, shape_obt))


def test_safe_factory(simple_volume):

    vp = VolumeProjector.safe_factory(simple_volume)

    shape_exp = [16, 17, 16]
    assert(np.allclose(shape_exp, vp.view_volume.GetSize()))
    assert(vp.view_volume.GetPixel(8, 9, 8) == simple_volume.GetPixel(5, 5, 4))
