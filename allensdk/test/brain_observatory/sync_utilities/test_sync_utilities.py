import pytest
import numpy as np

from allensdk.brain_observatory import sync_utilities as su


@pytest.mark.parametrize('vs_times, expected', [
    [[0.016, 0.033, 0.051, 0.067, 3.0], [0.016, 0.033, 0.051, 0.067]]
])
def test_trim_discontiguous_vsyncs(vs_times, expected):
    obtained = su.trim_discontiguous_times(vs_times)
    assert np.allclose(obtained, expected)