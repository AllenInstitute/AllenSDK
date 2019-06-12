from unittest import mock

import pytest
import numpy as np


from allensdk.brain_observatory.ecephys.align_timestamps.probe_synchronizer import (
    ProbeSynchronizer,
)


def get_test_barcodes():

    master_barcode_times = np.linspace(0, 30, 10)
    master_barcodes = np.arange(0, 11)

    probe_barcode_times = np.linspace(0, 30, 10) * 0.5 + 1
    probe_barcodes = np.arange(0, 11)

    min_time = 0
    max_time = 30

    return (
        master_barcode_times,
        master_barcodes,
        probe_barcode_times,
        probe_barcodes,
        min_time,
        max_time,
    )


@pytest.fixture
def synchronizer():

    local_sampling_rate = 4.0
    probe_start_index = 0

    mbt, mb, pbt, pb, min_time, max_time = get_test_barcodes()

    result = ProbeSynchronizer.compute(
        mbt, mb, pbt, pb, min_time, max_time, probe_start_index, local_sampling_rate
    )

    return result


@pytest.mark.parametrize(
    "samples,sync_condition,expected",
    [
        [
            np.arange(10, dtype="float"),
            "master",
            np.arange(10, dtype="float") / 2.0 - 2,
        ],
        [np.arange(10, dtype="float"), "probe", np.arange(10, dtype="float") / 4.0],
        [
            np.arange(10, dtype="float"),
            "salmon",
            np.arange(10, dtype="float") / 2.0 - 2,
        ],
    ],
)
def test_call(synchronizer, samples, sync_condition, expected):

    if sync_condition in ("master", "probe"):
        obtained = synchronizer(samples, sync_condition=sync_condition)
        # print(obtained)
        assert np.allclose(obtained, expected)

    else:
        with pytest.raises(ValueError):
            synchronizer(samples, sync_condition=sync_condition)


def test_sampling_rate_scale(synchronizer):

    assert synchronizer.sampling_rate_scale == 0.5
