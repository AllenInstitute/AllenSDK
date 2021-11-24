import pytest
from unittest.mock import create_autospec

import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects.running_speed.running_processing import (  # noqa: E501
    get_running_df
)
from allensdk.brain_observatory.behavior.data_objects import (
    RunningAcquisition, StimulusTimestamps
)


@pytest.mark.parametrize(
    "dict_repr, returned_running_acq_df, expected_running_acq_df",
    [
        (
            # dict_repr
            {
                "behavior_stimulus_file": "mock_stimulus_file.pkl"
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
    mock_stimulus_file = create_autospec(StimulusFile)
    mock_stimulus_timestamps = create_autospec(StimulusTimestamps)
    mock_get_running_df = create_autospec(get_running_df)

    mock_get_running_df.return_value = returned_running_acq_df

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_acquisition.StimulusFile",
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
        obt = RunningAcquisition.from_json(dict_repr)

    mock_stimulus_file.from_json.assert_called_once_with(dict_repr)
    mock_stimulus_file_instance = mock_stimulus_file.from_json(dict_repr)
    assert obt._stimulus_file == mock_stimulus_file_instance

    mock_stimulus_timestamps.from_json.assert_called_once_with(dict_repr)
    mock_stimulus_timestamps_instance = mock_stimulus_timestamps.from_json(
        dict_repr
    )
    assert obt._stimulus_timestamps == mock_stimulus_timestamps_instance

    mock_get_running_df.assert_called_once_with(
        data=mock_stimulus_file_instance.data,
        time=mock_stimulus_timestamps_instance.value,
    )

    pd.testing.assert_frame_equal(obt.value, expected_running_acq_df)


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
            "RunningAcquisition DataObject lacks information about",
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
            "RunningAcquisition DataObject lacks information about",
            # expected
            None
        ),
    ]
)
def test_running_acquisition_to_json(
    stimulus_file, stimulus_file_to_json_ret,
    stimulus_timestamps, stimulus_timestamps_to_json_ret, raises, expected
):
    if stimulus_file is not None:
        stimulus_file.to_json.return_value = stimulus_file_to_json_ret
    if stimulus_timestamps is not None:
        stimulus_timestamps.to_json.return_value = (
            stimulus_timestamps_to_json_ret
        )

    running_acq = RunningAcquisition(
        running_acquisition=None,
        stimulus_file=stimulus_file,
        stimulus_timestamps=stimulus_timestamps
    )

    if raises:
        with pytest.raises(RuntimeError, match=raises):
            _ = running_acq.to_json()
    else:
        obt = running_acq.to_json()
        assert obt == expected


@pytest.mark.parametrize(
    "behavior_session_id, ophys_experiment_id, "
    "returned_running_acq_df, expected_running_acq_df",
    [
        (
            # behavior_session_id
            12345,
            # ophys_experiment_id
            None,
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
        (
            # behavior_session_id
            1234,
            # ophys_experiment_id
            5678,
            # returned_running_acq_df
            pd.DataFrame(
                {
                    "timestamps": [2, 4],
                    "speed": [6, 8],
                    "dx": [10, 12],
                    "v_sig": [14, 16],
                    "v_in": [18, 20]
                }
            ).set_index("timestamps"),
            # expected_running_acq_df
            pd.DataFrame(
                {
                    "timestamps": [2, 4],
                    "dx": [10, 12],
                    "v_sig": [14, 16],
                    "v_in": [18, 20]
                }
            ).set_index("timestamps")
        )
    ]
)
def test_running_acquisition_from_lims(
    monkeypatch, behavior_session_id, ophys_experiment_id,
    returned_running_acq_df, expected_running_acq_df
):
    mock_db_conn = create_autospec(PostgresQueryMixin, instance=True)

    mock_stimulus_file = create_autospec(StimulusFile)
    mock_stimulus_timestamps = create_autospec(StimulusTimestamps)
    mock_get_running_df = create_autospec(get_running_df)

    mock_get_running_df.return_value = returned_running_acq_df

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".running_speed.running_acquisition.StimulusFile",
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
        obt = RunningAcquisition.from_lims(
            mock_db_conn, behavior_session_id, ophys_experiment_id
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

    mock_get_running_df.assert_called_once_with(
        data=mock_stimulus_file_instance.data,
        time=mock_stimulus_timestamps_instance.value,
    )

    pd.testing.assert_frame_equal(
        obt.value, expected_running_acq_df, check_like=True
    )


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
