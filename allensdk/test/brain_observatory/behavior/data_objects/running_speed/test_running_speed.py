import pytest
from unittest.mock import create_autospec

import pandas as pd

from allensdk.core.exceptions import DataFrameIndexError
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df
)
from allensdk.brain_observatory.behavior.data_objects import (
    RunningSpeed, StimulusTimestamps
)


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

    mock_stimulus_file_instance = create_autospec(StimulusFile, instance=True)
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
    mock_stimulus_file = create_autospec(StimulusFile)
    mock_stimulus_timestamps = create_autospec(StimulusTimestamps)
    mock_get_running_speed_df = create_autospec(get_running_df)

    mock_get_running_speed_df.return_value = returned_running_df

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_speed.StimulusFile",
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
        obt = RunningSpeed.from_json(dict_repr, filtered, zscore_threshold)

    mock_stimulus_file.from_json.assert_called_once_with(dict_repr)
    mock_stimulus_file_instance = mock_stimulus_file.from_json(dict_repr)
    assert obt._stimulus_file == mock_stimulus_file_instance

    mock_stimulus_timestamps.from_json.assert_called_once_with(dict_repr)
    mock_stimulus_timestamps_instance = mock_stimulus_timestamps.from_json(
        dict_repr
    )
    assert obt._stimulus_timestamps == mock_stimulus_timestamps_instance

    mock_get_running_speed_df.assert_called_once_with(
        data=mock_stimulus_file_instance.data,
        time=mock_stimulus_timestamps_instance.value,
        lowpass=filtered,
        zscore_threshold=zscore_threshold
    )

    assert obt._filtered == filtered
    pd.testing.assert_frame_equal(obt.value, expected_running_df)


@pytest.mark.parametrize(
    "stimulus_file, stimulus_file_to_json_ret, "
    "stimulus_timestamps, stimulus_timestamps_to_json_ret, raises, expected",
    [
        # Test to_json with both stimulus_file and sync_file
        (
            # stimulus_file
            create_autospec(StimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # stimulus_timestamps
            create_autospec(StimulusTimestamps, instance=True),
            # stimulus_timestamps_to_json_ret
            {"sync_file": "sync.h5"},
            # raises
            False,
            # expected
            {"behavior_stimulus_file": "stim.pkl", "sync_file": "sync.h5"}
        ),
        # Test to_json without stimulus_file
        (
            # stimulus_file
            None,
            # stimulus_file_to_json_ret
            None,
            # stimulus_timestamps
            create_autospec(StimulusTimestamps, instance=True),
            # stimulus_timestamps_to_json_ret
            {"sync_file": "sync.h5"},
            # raises
            "RunningSpeed DataObject lacks information about",
            # expected
            None
        ),
        # Test to_json without stimulus_timestamps
        (
            # stimulus_file
            create_autospec(StimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # stimulus_timestamps_to_json_ret
            None,
            # sync_file_to_json_ret
            None,
            # raises
            "RunningSpeed DataObject lacks information about",
            # expected
            None
        ),
    ]
)
def test_running_speed_to_json(
    stimulus_file, stimulus_file_to_json_ret,
    stimulus_timestamps, stimulus_timestamps_to_json_ret, raises, expected
):
    if stimulus_file is not None:
        stimulus_file.to_json.return_value = stimulus_file_to_json_ret
    if stimulus_timestamps is not None:
        stimulus_timestamps.to_json.return_value = (
            stimulus_timestamps_to_json_ret
        )

    running_speed = RunningSpeed(
        running_speed=None,
        stimulus_file=stimulus_file,
        stimulus_timestamps=stimulus_timestamps
    )

    if raises:
        with pytest.raises(RuntimeError, match=raises):
            _ = running_speed.to_json()
    else:
        obt = running_speed.to_json()
        assert obt == expected


@pytest.mark.parametrize("behavior_session_id", [12345, 1234])
@pytest.mark.parametrize("filtered", [True, False])
@pytest.mark.parametrize("zscore_threshold", [1.0, 4.2])
@pytest.mark.parametrize(
    "returned_running_df, expected_running_df",
    [
        (
            # returned_running_df
            pd.DataFrame(
                {"timestamps": [1, 2], "speed": [3, 4]}
            ).set_index("timestamps"),
            # expected_running_df
            pd.DataFrame(
                {"timestamps": [1, 2], "speed": [3, 4]}
            ),
        ),
        (
            # returned_running_df
            pd.DataFrame(
                {"timestamps": [1, 2], "speed": [3, 4]}
            ).set_index("timestamps"),
            # expected_running_df
            pd.DataFrame(
                {"timestamps": [1, 2], "speed": [3, 4]}
            ),
        )
    ]
)
def test_running_speed_from_lims(
    monkeypatch, behavior_session_id, returned_running_df,
    expected_running_df, filtered, zscore_threshold
):
    mock_db_conn = create_autospec(PostgresQueryMixin, instance=True)

    mock_stimulus_file = create_autospec(StimulusFile)
    mock_stimulus_timestamps = create_autospec(StimulusTimestamps)
    mock_get_running_speed_df = create_autospec(get_running_df)
    mock_get_running_speed_df.return_value = returned_running_df

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_speed.StimulusFile",
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
        obt = RunningSpeed.from_lims(
            mock_db_conn, behavior_session_id, filtered,
            zscore_threshold
        )

    mock_stimulus_file.from_lims.assert_called_once_with(
        mock_db_conn, behavior_session_id
    )
    mock_stimulus_file_instance = mock_stimulus_file.from_lims(
        mock_db_conn, behavior_session_id
    )
    assert obt._stimulus_file == mock_stimulus_file_instance

    mock_stimulus_timestamps.from_stimulus_file.assert_called_once_with(
        mock_stimulus_file_instance
    )
    mock_stimulus_timestamps_instance = mock_stimulus_timestamps.\
        from_stimulus_file(stimulus_file=mock_stimulus_file)
    assert obt._stimulus_timestamps == mock_stimulus_timestamps_instance

    mock_get_running_speed_df.assert_called_once_with(
        data=mock_stimulus_file_instance.data,
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
