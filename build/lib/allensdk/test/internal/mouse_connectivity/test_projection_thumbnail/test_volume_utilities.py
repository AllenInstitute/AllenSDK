import pytest
import numpy as np
import SimpleITK as sitk

import allensdk.internal.mouse_connectivity.projection_thumbnail.volume_utilities as vol


@pytest.fixture
def empty_image():
    img = sitk.Image(2, 3, 4, sitk.sitkFloat64)
    return img


@pytest.fixture
def even_image():
    arr = np.zeros([12, 14, 16])
    arr[6, 7, 8] = 1   
    arr[6, 7, 8] = 1 
    img = sitk.GetImageFromArray(arr)
    return img


def test_sitk_get_image_parameters(empty_image):
  
    sp, sz, og = vol.sitk_get_image_parameters(empty_image)
  
    assert(np.allclose(sp, [1, 1, 1]))
    assert(np.allclose(sz, [2, 3, 4]))
    assert(np.allclose(og, [0, 0, 0]))


def test_sitk_get_center_even(even_image):

    center = vol.sitk_get_center(even_image)
    assert(np.allclose(center, [7.5, 6.5, 5.5]))


def test_sitk_get_center_odd(empty_image):
    obt = vol.sitk_get_center(empty_image)
    assert(np.allclose(obt, [0.5, 1, 1.5]))


@pytest.mark.parametrize('size,exp', [([2, 3], [0, 1]), ([3, 3], [1, 1])])
def test_sitk_get_size_parity(size, exp):

    image = sitk.Image(size[0], size[1], sitk.sitkUInt8)
    obt = vol.sitk_get_size_parity(image)
  
    assert(np.allclose(exp, obt))


@pytest.mark.parametrize('shape,exp', [([1, 1, 1], np.sqrt(3)), 
                                       ([2, 4], np.sqrt(20))])
def test_sitk_get_diagonal_length(shape, exp):

    img = sitk.GetImageFromArray(np.zeros(shape))
    obt = vol.sitk_get_diagonal_length(img)

    assert(exp == obt)


def test_sitk_paste_into_center_even():

    smaller = sitk.GetImageFromArray(np.eye(2))
    larger = sitk.GetImageFromArray(np.zeros((4, 4)))

    obt = vol.sitk_paste_into_center(smaller, larger)
    obt = sitk.GetArrayFromImage(obt)

    exp = np.zeros((4, 4))
    exp[1, 1] = 1
    exp[2, 2] = 1

    assert(np.allclose(obt, exp))


def test_sitk_paste_into_center_odd():

    smaller = sitk.GetImageFromArray(np.eye(3, 3))
    larger = sitk.GetImageFromArray(np.zeros((5, 5)))

    obt = vol.sitk_paste_into_center(smaller, larger)
    obt = sitk.GetArrayFromImage(obt)

    print(obt)

    exp = np.zeros((5, 5))
    exp[1, 1] = 1
    exp[2, 2] = 1
    exp[3, 3] = 1

    assert(np.allclose(exp, obt))
