import pytest


import SimpleITK as sitk
import numpy as np
import matplotlib as mpl

from allensdk.internal.mouse_connectivity.projection_thumbnail import visualization_utilities as vis


@pytest.fixture
def example_volume():
    arr = np.arange(5*6*7, dtype=np.float64).reshape([5, 6, 7])
    return sitk.GetImageFromArray(arr)


@pytest.fixture
def discrete_cmap():
    red = np.arange(255)
    green = np.arange(255)[::-1]
    blue = np.zeros(256)
    return [[r, g, b] for r, g, b in zip(red, green, blue)]


def test_convert_discrete_colormap(discrete_cmap):
    cmap = vis.convert_discrete_colormap(discrete_cmap)

    obt = cmap(0.75)
    exp = [3 * 255 / 4.0, 1 * 255 / 4.0, 0.0, 1.0]


def test_sitk_safe_ln(example_volume):
    
    obt = sitk.GetArrayFromImage(vis.sitk_safe_ln(example_volume))
    arr = sitk.GetArrayFromImage(example_volume)
    
    arr = np.log(arr)
    arr[0, 0, 0] = np.log(10**-10)
    
    print(obt)
    print(arr)

    assert(np.allclose(arr, obt))


def test_normalize_intensity(example_volume):
    
    obt = vis.normalize_intensity(example_volume, 2, 4, 50, 100)
    obt = sitk.GetArrayFromImage(obt)

    assert(75 == obt[0, 0, 3])


def test_blend():
    
    images = [np.eye(2), np.array([[1, 2], [3, 4]])]
    weights = [np.fliplr(np.eye(2)), [[0, 0], [0, 1]]]

    exp = [[1, 0], [0, 4]]
    obt = vis.blend(images, weights)
    assert(np.allclose(exp, obt))
