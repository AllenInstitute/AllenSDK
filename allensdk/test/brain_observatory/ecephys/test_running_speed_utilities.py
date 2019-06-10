import pytest
import numpy as np


from allensdk.brain_observatory.ecephys.extract_running_speed import utilities as ut


@pytest.mark.parametrize('degrees', [
    [12],
    np.linspace(0, np.pi, 1000),
    np.linspace(0, np.pi, 1000)
])
def test_degrees_to_radians(degrees):

    obtained = ut.degrees_to_radians(degrees)
    roundtrip = obtained  * 180 / np.pi

    assert np.allclose(degrees, roundtrip)


@pytest.mark.parametrize('angular,radius', [
    [12, 1],
    [np.arange(20), 4] 
])
def test_angular_to_linear_velocity(angular, radius):

    obtained = ut.angular_to_linear_velocity(angular, radius)
    roundtrip = obtained / radius

    assert np.allclose(angular, roundtrip)
