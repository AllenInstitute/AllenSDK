import pytest
from unittest.mock import create_autospec

import pandas as pd

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df
)
from allensdk.brain_observatory.behavior.data_objects import (
    RunningAcquisition, StimulusTimestamps
)


def test_nonzero_monitor_delay_acq():
    """
    Test that RunningAcquisition throws an exception if instantiated
    with a timestamps object that has non-zero monitor_delay
    """
    class OtherTimestamps(object):
        monitor_delay = 0.01
        value = 0.0

    with pytest.raises(RuntimeError,
                       match="should be no monitor delay"):

        RunningAcquisition(
            running_acquisition=4.0,
            stimulus_file=None,
            stimulus_timestamps=OtherTimestamps())


@pytest.mark.parametrize(
    "dict_repr, returned_running_acq_df, expected_running_acq_df",
    [
        (
            # dict_repr
            {
                "behavior_stimulus_file": "mock_stimulus_file.pkl",
                "monitor_delay": 0.0
            },
            # returned_running_acq_df
            pd.DataFrame(
                {
                    "timestamps": [1, 2],
                    "speed": [3, 4],
                    "dx": [5, 6],
                    "v_sig": [7, 8],
                    "v_in": [9, 10]
                }
            ).set_index("timestamps"),
            # expected_running_acq_df
            pd.DataFrame(
                {
                    "timestamps": [1, 2],
                    "dx": [5, 6],
                    "v_sig": [7, 8],
                    "v_in": [9, 10]
                }
            ).set_index("timestamps")
        ),
    ]
)
def test_running_acquisition_from_json(
    monkeypatch, dict_repr, returned_running_acq_df, expected_running_acq_df
):
    mock_stimulus_file = create_autospec(BehaviorStimulusFile)
    mock_stimulus_timestamps = create_autospec(StimulusTimestamps)

    class DummyTimestamps(object):
        monitor_delay = 0.0
        value = 0.0
    dummy_ts = DummyTimestamps()
    mock_stimulus_timestamps.from_stimulus_file.return_value = dummy_ts
    mock_stimulus_timestamps.from_json.return_value = dummy_ts

    mock_get_running_df = create_autospec(get_running_df)

    mock_get_running_df.return_value = returned_running_acq_df

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_acquisition.BehaviorStimulusFile",
            mock_stimulus_file
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_acquisition.StimulusTimestamps",
            mock_stimulus_timestamps
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_acquisition.get_running_df",
            mock_get_running_df
        )
        obt = RunningAcquisition.from_stimulus_file(
                    behavior_stimulus_file=mock_stimulus_file)

    mock_stimulus_file_instance = mock_stimulus_file.from_json(dict_repr)

    mock_stimulus_timestamps_instance = \
        mock_stimulus_timestamps.from_stimulus_file(
            stimulus_file=mock_stimulus_file_instance,
            monitor_delay=0.0
        )
    assert obt._stimulus_timestamps == mock_stimulus_timestamps_instance

    pd.testing.assert_frame_equal(obt.value, expected_running_acq_df)


# Fixtures:
# nwbfile:
#   test/brain_observatory/behavior/conftest.py
# data_object_roundtrip_fixture:
#   test/brain_observatory/behavior/data_objects/conftest.py
@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("running_acq_data", [
    (
        # expected_running_acq_df
        pd.DataFrame(
            {
                "timestamps": [2.0, 4.0],
                "dx": [10.0, 12.0],
                "v_sig": [14.0, 16.0],
                "v_in": [18.0, 20.0]
            }
        ).set_index("timestamps")
    ),
])
def test_running_acquisition_nwb_roundtrip(
    nwbfile, data_object_roundtrip_fixture, roundtrip, running_acq_data
):
    running_acq = RunningAcquisition(running_acquisition=running_acq_data)
    nwbfile = running_acq.to_nwb(nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(nwbfile, RunningAcquisition)
    else:
        obt = RunningAcquisition.from_nwb(nwbfile)

    pd.testing.assert_frame_equal(
        obt.value, running_acq_data, check_like=True
    )
