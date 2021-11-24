from unittest.mock import create_autospec, PropertyMock

import numpy as np
import pytest

from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .stimulus_timestamps.timestamps_processing import (
        get_behavior_stimulus_timestamps, get_ophys_stimulus_timestamps)
from allensdk.internal.brain_observatory.time_sync import OphysTimeAligner


@pytest.mark.parametrize("pkl_data, expected", [
    # Extremely basic test case
    (
        # pkl_data
        {
            "items": {
                "behavior": {
                    "intervalsms": np.array([
                        1000, 1001, 1002, 1003, 1004, 1005
                    ])
                }
            }
        },
        # expected
        np.array([
            0.0, 1.0, 2.001, 3.003, 4.006, 5.01, 6.015
        ])
    ),
    # More realistic test case
    (
        # pkl_data
        {
            "items": {
                "behavior": {
                    "intervalsms": np.array([
                        16.5429, 16.6685, 16.66580001, 16.70569999,
                        16.6668,
                        16.69619999, 16.655, 16.6805, 16.75940001, 16.6831
                    ])
                }
            }
        },
        # expected
        np.array([
            0.0, 0.0165429, 0.0332114, 0.0498772, 0.0665829, 0.0832497,
            0.0999459, 0.1166009, 0.1332814, 0.1500408, 0.1667239
        ])
    )
])
def test_get_behavior_stimulus_timestamps(pkl_data, expected):
    obt = get_behavior_stimulus_timestamps(pkl_data)
    assert np.allclose(obt, expected)


@pytest.mark.parametrize("sync_path, expected_timestamps", [
    ("/tmp/mock_sync_file.h5", [1, 2, 3]),
])
def test_get_ophys_stimulus_timestamps(
        monkeypatch, sync_path, expected_timestamps
):
    mock_ophys_time_aligner = create_autospec(OphysTimeAligner)
    mock_aligner_instance = mock_ophys_time_aligner.return_value
    property_mock = PropertyMock(
        return_value=(expected_timestamps, "ignored_return_val")
    )
    type(mock_aligner_instance).clipped_stim_timestamps = property_mock

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.timestamps_processing"
            ".OphysTimeAligner",
            mock_ophys_time_aligner
        )
        obt = get_ophys_stimulus_timestamps(sync_path)

    mock_ophys_time_aligner.assert_called_with(sync_file=sync_path)
    property_mock.assert_called_once()
    assert np.allclose(obt, expected_timestamps)
