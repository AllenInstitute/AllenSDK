import pytest

import SimpleITK as sitk
import numpy as np

from allensdk.internal.mouse_connectivity.projection_thumbnail import projection_functions as prf


@pytest.fixture
def example_volume():
    array = np.arange(8, dtype=np.float64).reshape([2, 2, 2])
    return sitk.GetImageFromArray(array)


def test_convert_axis(): # :)

    obt = prf.convert_axis(2)
    assert(obt == 0)


def test_max_projection(example_volume):

    max_obt, depth_obt = prf.max_projection(example_volume, 2)

    depth_exp = np.zeros([2, 2]) + 1
    max_exp = np.array([[4, 5], [6, 7]])

    assert(np.allclose(depth_exp, depth_obt))
    assert(np.allclose(max_exp, max_obt))


def test_template_projection(example_volume):

    obt = prf.template_projection(example_volume, 2, 1, 1)
    exp = np.array([[16, 25], [20, 5]])

