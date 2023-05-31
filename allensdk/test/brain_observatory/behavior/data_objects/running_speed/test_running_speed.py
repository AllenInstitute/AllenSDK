import pytest
from unittest.mock import create_autospec

import pandas as pd

from allensdk.core.exceptions import DataFrameIndexError
from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df
)
from allensdk.brain_observatory.behavior.data_objects import (
    RunningSpeed, StimulusTimestamps
)


class DummyTimestamps(object):
    """
    A class meant to mock the StimulusTimestamps API by providing
    monitor_delay=0.0, and value=0.0
    """
    monitor_delay = 0.0
    value = 0.0


def test_nonzero_monitor_delay_speed():
    """
    Test that RunningSpeed throws an exception if instantiated
    with a timestamps object that has non-zero monitor_delay
    """
    class OtherTimestamps(object):
        monitor_delay = 0.01
        value = 0.0

    with pytest.raises(RuntimeError,
                       match="should be no monitor delay"):

        RunningSpeed(
            running_speed=pd.DataFrame({'speed': [4.0]}),
            stimulus_file=None,
            sync_file=None,
            stimulus_timestamps=OtherTimestamps())


@pytest.mark.parametrize("filtered", [True, False])
@pytest.mark.parametrize("zscore_threshold", [1.0, 4.2])
@pytest.mark.parametrize("returned_running_df, expected_running_df, raises", [
    # Test basic case
    (
        # returned_running_df
        pd.DataFrame({
            "timestamps": [2, 4, 6, 8],
            "speed": [1, 2, 3, 4]
        }).set_index("timestamps"),
        # expected_running_df
        pd.DataFrame({
            "timestamps": [2, 4, 6, 8],
            "speed": [1, 2, 3, 4]
        }),
        # raises
        False
    ),
    # Test when returned dataframe lacks "timestamps" as index
    (
        # returned_running_df
        pd.DataFrame({
            "timestamps": [2, 4, 6, 8],
            "speed": [1, 2, 3, 4]
        }).set_index("speed"),
        # expected_running_df
        None,
        # raises
        "Expected running_data_df index to be named 'timestamps'"
    ),
])
def test_get_running_speed_df(
    monkeypatch, returned_running_df, filtered, zscore_threshold,
    expected_running_df, raises
):

    mock_stimulus_file_instance = create_autospec(
                                      BehaviorStimulusFile,
                                      instance=True)
    mock_stimulus_timestamps_instance = create_autospec(
        StimulusTimestamps, instance=True
    )
    mock_get_running_speed_df = create_autospec(get_running_df)
    mock_get_running_speed_df.return_value = returned_running_df

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_speed.get_running_df",
            mock_get_running_speed_df
        )

        if raises:
            with pytest.raises(DataFrameIndexError, match=raises):
                _ = RunningSpeed._get_running_speed_df(
                    mock_stimulus_file_instance,
                    mock_stimulus_timestamps_instance,
                    filtered, zscore_threshold
                )
        else:
            obt = RunningSpeed._get_running_speed_df(
                mock_stimulus_file_instance,
                mock_stimulus_timestamps_instance,
                filtered, zscore_threshold
            )

            pd.testing.assert_frame_equal(obt, expected_running_df)

    mock_get_running_speed_df.assert_called_once_with(
        data=mock_stimulus_file_instance.data,
        time=mock_stimulus_timestamps_instance.value,
        lowpass=filtered,
        zscore_threshold=zscore_threshold
    )


@pytest.mark.parametrize("filtered", [True, False])
@pytest.mark.parametrize("zscore_threshold", [1.0, 4.2])
@pytest.mark.parametrize(
    "dict_repr, returned_running_df, expected_running_df",
    [
        (
            # dict_repr
            {
                "behavior_stimulus_file": "mock_stimulus_file.pkl"
            },
            # returned_running_df
            pd.DataFrame(
                {"timestamps": [1, 2], "speed": [3, 4]}
            ).set_index("timestamps"),
            # expected_running_df
            pd.DataFrame(
                {"timestamps": [1, 2], "speed": [3, 4]}
            ),
        ),
    ]
)
def test_running_speed_from_json(
    monkeypatch, dict_repr, returned_running_df, expected_running_df,
    filtered, zscore_threshold
):
    mock_stimulus_file = create_autospec(BehaviorStimulusFile)
    mock_stimulus_timestamps = create_autospec(StimulusTimestamps)

    dummy_ts = DummyTimestamps()
    mock_stimulus_timestamps.from_stimulus_file.return_value = dummy_ts
    mock_stimulus_timestamps.from_json.return_value = dummy_ts

    mock_get_running_speed_df = create_autospec(get_running_df)

    mock_get_running_speed_df.return_value = returned_running_df

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_speed.BehaviorStimulusFile",
            mock_stimulus_file
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_speed.StimulusTimestamps",
            mock_stimulus_timestamps
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_speed.get_running_df",
            mock_get_running_speed_df
        )

        obt = RunningSpeed.from_stimulus_file(
                    behavior_stimulus_file=mock_stimulus_file,
                    filtered=filtered,
                    zscore_threshold=zscore_threshold)

    mock_stimulus_file_instance = mock_stimulus_file.from_json(dict_repr)

    mock_stimulus_timestamps_instance = \
        mock_stimulus_timestamps.from_stimulus_file(
                stimulus_file=mock_stimulus_file_instance,
                monitor_delay=0.0
        )
    assert obt._stimulus_timestamps == mock_stimulus_timestamps_instance

    mock_get_running_speed_df.assert_called_once_with(
        data=mock_stimulus_file.data,
        time=mock_stimulus_timestamps_instance.value,
        lowpass=filtered,
        zscore_threshold=zscore_threshold
    )

    assert obt._filtered == filtered
    pd.testing.assert_frame_equal(obt.value, expected_running_df)


# Fixtures:
# nwbfile:
#   test/brain_observatory/behavior/conftest.py
# data_object_roundtrip_fixture:
#   test/brain_observatory/behavior/data_objects/conftest.py
@pytest.mark.parametrize("roundtrip", [True, False])
@pytest.mark.parametrize("filtered", [True, False])
@pytest.mark.parametrize("running_speed_data", [
    (pd.DataFrame({"timestamps": [3.0, 4.0], "speed": [5.0, 6.0]})),
])
def test_running_speed_nwb_roundtrip(
    nwbfile, data_object_roundtrip_fixture, roundtrip, running_speed_data,
    filtered
):
    running_speed = RunningSpeed(
        running_speed=running_speed_data, filtered=filtered
    )
    nwbfile = running_speed.to_nwb(nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile, RunningSpeed, filtered=filtered
        )
    else:
        obt = RunningSpeed.from_nwb(nwbfile, filtered=filtered)

    pd.testing.assert_frame_equal(obt.value, running_speed_data)
