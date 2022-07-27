from unittest.mock import create_autospec, PropertyMock

import numpy as np
import pytest

from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .stimulus_timestamps.timestamps_processing import (
        get_behavior_stimulus_timestamps,
        get_ophys_stimulus_timestamps,
        get_frame_indices)

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


@pytest.mark.parametrize(
    "frame_timestamps, event_timestamps, expected_indices",
    [(np.arange(0, 1, 0.11),
      np.array([0.22, 0.13, 0.32, 0.77]),
      np.array([2, 1, 2, 7])),
     (np.arange(0, 1, 0.11),
      np.array([-0.1, 2.1, 0.55, 0.42]),
      np.array([0, 9, 5, 3])),
     (np.array([0.11, 0.11, 0.33, 0.33, 0.44]),
      np.array([0.05, 0.12, 0.22, 0.33, 0.77]),
      np.array([0, 1, 1, 2, 4]))])
def test_get_frame_indices(
        frame_timestamps,
        event_timestamps,
        expected_indices):
    """
    Test that get_frame_indices correctly associates events
    with frames
    """
    actual = get_frame_indices(
                frame_timestamps=frame_timestamps,
                event_timestamps=event_timestamps)

    np.testing.assert_array_equal(actual, expected_indices)


def test_get_frame_indices_error():
    """
    Test that a ValueError is raised when unsorted
    frame_timestamps are passed into get_frame_indices
    """

    frame_timestamps = np.array([0.1, 0.4, 0.3])
    event_timestamps = np.array([0.11, 0.22])
    with pytest.raises(ValueError,
                       match="frame_timestamps are not in ascending order"):
        get_frame_indices(
            frame_timestamps=frame_timestamps,
            event_timestamps=event_timestamps)
