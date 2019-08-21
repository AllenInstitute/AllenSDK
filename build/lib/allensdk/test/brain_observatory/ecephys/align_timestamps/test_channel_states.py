from unittest import mock

import pytest
import numpy as np

from allensdk.brain_observatory.ecephys.align_timestamps import channel_states as cs


@pytest.mark.parametrize(
    "sample_frequency,events,times,times_exp,codes_exp",
    [
        [
            1,
            np.array([1, 1, 1, -1, -1, -1]),
            np.array([30, 50, 50.08, 31, 50.04, 50.12]),
            [50],
            [3],
        ]
    ],
)
def test_extract_barcodes_from_states(
    sample_frequency, events, times, times_exp, codes_exp
):

    times, codes = cs.extract_barcodes_from_states(events, times, sample_frequency)

    assert np.allclose(times, times_exp)
    assert np.allclose(codes, codes_exp)
