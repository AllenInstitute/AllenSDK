import numpy as np
import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.running_processing import (
    get_running_df, calc_deriv, deg_to_dist)


@pytest.fixture
def running_data():
    return {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": np.array([0., 0.8444478, 0.7076058, 1.4225141,
                                        1.5040479]),
                        "vsig": [3.460190074169077, 3.4692217108095065,
                                 3.4808338150614873, 3.5014775559538975,
                                 3.5259919982636347],
                        "vin": [4.996858536847867, 4.99298783543054,
                                4.995568303042091, 4.996858536847867,
                                5.00201947207097],
                    }]}}}


@pytest.fixture
def timestamps():
    return np.array([0., 0.01670847, 0.03336808, 0.05002418, 0.06672007])


@pytest.mark.parametrize(
    "x,time,expected", [
        ([1.0, 1.0], [1.0, 2.0], [0.0, 0.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
        ([1.0, 2.0, 3.0], [1.0, 4.0, 6.0], [1/3, ((1/3)+0.5)/2, 0.5])
    ]
)
def test_calc_deriv(x, time, expected):
    assert np.all(calc_deriv(x, time) == expected)


@pytest.mark.parametrize(
    "speed,expected", [
        (np.array([1.0]), [0.09605128650142128]),
        (np.array([0., 2.0]), [0., 2.0 * 0.09605128650142128])
    ]
)
def test_deg_to_dist(speed, expected):
    assert np.all(np.allclose(deg_to_dist(speed), expected))


def test_get_running_df(running_data, timestamps):
    expected = pd.DataFrame(
        {'speed': {
            0.0: 4.0677840296488785,
            0.01670847: 4.468231641421186,
            0.03336808: 4.869192250359061,
            0.05002418: 4.47027713320348,
            0.06672007: 4.070849018882336},
         'dx': {
            0.0: 0.0,
            0.01670847: 0.8444478,
            0.03336808: 0.7076058,
            0.05002418: 1.4225141,
            0.06672007: 1.5040479},
         'v_sig': {
            0.0: 3.460190074169077,
            0.01670847: 3.4692217108095065,
            0.03336808: 3.4808338150614873,
            0.05002418: 3.5014775559538975,
            0.06672007: 3.5259919982636347},
         'v_in': {
            0.0: 4.996858536847867,
            0.01670847: 4.99298783543054,
            0.03336808: 4.995568303042091,
            0.05002418: 4.996858536847867,
            0.06672007: 5.00201947207097}})
    expected.index.name = "timestamps"

    pd.testing.assert_frame_equal(expected,
                                  get_running_df(running_data, timestamps))
