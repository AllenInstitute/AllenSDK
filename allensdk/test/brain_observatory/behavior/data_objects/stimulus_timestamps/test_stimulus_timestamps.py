from pathlib import Path

import pytest
from unittest.mock import create_autospec

import numpy as np

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import (
    StimulusFile, SyncFile
)
from allensdk.brain_observatory.behavior.data_objects.timestamps\
    .stimulus_timestamps.timestamps_processing import (
        get_behavior_stimulus_timestamps, get_ophys_stimulus_timestamps)
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps


@pytest.mark.parametrize("dict_repr, has_pkl, has_sync", [
    # Test where input json only has "behavior_stimulus_file"
    (
        # dict_repr
        {
            "behavior_stimulus_file": "mock_stimulus_file.pkl"
        },
        # has_pkl
        True,
        # has_sync
        False
    ),
    # Test where input json has both "behavior_stimulus_file" and "sync_file"
    (
        # dict_repr
        {
            "behavior_stimulus_file": "mock_stimulus_file.pkl",
            "sync_file": "mock_sync_file.h5"
        },
        # has_pkl
        True,
        # has_sync
        True
    ),
])
def test_stimulus_timestamps_from_json(
    monkeypatch, dict_repr, has_pkl, has_sync
):
    mock_stimulus_file = create_autospec(StimulusFile)
    mock_sync_file = create_autospec(SyncFile)

    mock_get_behavior_stimulus_timestamps = create_autospec(
        get_behavior_stimulus_timestamps
    )
    mock_get_ophys_stimulus_timestamps = create_autospec(
        get_ophys_stimulus_timestamps
    )

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps.StimulusFile",
            mock_stimulus_file
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps.SyncFile",
            mock_sync_file
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps"
            ".get_behavior_stimulus_timestamps",
            mock_get_behavior_stimulus_timestamps
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps"
            ".get_ophys_stimulus_timestamps",
            mock_get_ophys_stimulus_timestamps
        )
        mock_stimulus_file_instance = mock_stimulus_file.from_json(dict_repr)
        ts_from_stim = StimulusTimestamps.from_stimulus_file(
            stimulus_file=mock_stimulus_file_instance)

        if has_pkl and has_sync:
            mock_sync_file_instance = mock_sync_file.from_json(dict_repr)
            ts_from_sync = StimulusTimestamps.from_sync_file(
                sync_file=mock_sync_file_instance)

    if has_pkl and has_sync:
        mock_get_ophys_stimulus_timestamps.assert_called_once_with(
            sync_path=mock_sync_file_instance.filepath
        )
        assert ts_from_sync._sync_file == mock_sync_file_instance
    else:
        assert ts_from_stim._stimulus_file == mock_stimulus_file_instance
        mock_get_behavior_stimulus_timestamps.assert_called_once_with(
            stimulus_pkl=mock_stimulus_file_instance.data
        )


def test_stimulus_timestamps_from_json2():
    dir = Path(__file__).parent.parent.resolve()
    test_data_dir = dir / 'test_data'
    sf_path = test_data_dir / 'stimulus_file.pkl'

    sf = StimulusFile.from_json(
        dict_repr={'behavior_stimulus_file': str(sf_path)})
    stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
        stimulus_file=sf)
    expected = np.array([0.016 * i for i in range(11)])
    assert np.allclose(expected, stimulus_timestamps.value)


def test_stimulus_timestamps_from_json3():
    """
    Test that StimulusTimestamps.from_stimulus_file
    just returns the sum of the intervalsms field in the
    behavior stimulus pickle file, padded with a zero at the
    first timestamp.
    """
    dir = Path(__file__).parent.parent.resolve()
    test_data_dir = dir / 'test_data'
    sf_path = test_data_dir / 'stimulus_file.pkl'

    sf = StimulusFile.from_json(
        dict_repr={'behavior_stimulus_file': str(sf_path)})

    sf._data['items']['behavior']['intervalsms'] = [0.1, 0.2, 0.3, 0.4]

    stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
        stimulus_file=sf)

    expected = np.array([0., 0.0001, 0.0003, 0.0006, 0.001])
    np.testing.assert_array_almost_equal(stimulus_timestamps.value,
                                         expected,
                                         decimal=10)


@pytest.mark.parametrize(
    "stimulus_file, stimulus_file_to_json_ret, "
    "sync_file, sync_file_to_json_ret, raises, expected",
    [
        # Test to_json with both stimulus_file and sync_file
        (
            # stimulus_file
            create_autospec(StimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # sync_file
            create_autospec(SyncFile, instance=True),
            # sync_file_to_json_ret
            {"sync_file": "sync.h5"},
            # raises
            False,
            # expected
            {"behavior_stimulus_file": "stim.pkl", "sync_file": "sync.h5"}
        ),
        # Test to_json with only stimulus_file
        (
            # stimulus_file
            create_autospec(StimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # sync_file
            None,
            # sync_file_to_json_ret
            None,
            # raises
            False,
            # expected
            {"behavior_stimulus_file": "stim.pkl"}
        ),
        # Test to_json without stimulus_file nor sync_file
        (
            # stimulus_file
            None,
            # stimulus_file_to_json_ret
            None,
            # sync_file
            None,
            # sync_file_to_json_ret
            None,
            # raises
            "StimulusTimestamps DataObject lacks information about",
            # expected
            None
        ),
    ]
)
def test_stimulus_timestamps_to_json(
    stimulus_file, stimulus_file_to_json_ret,
    sync_file, sync_file_to_json_ret, raises, expected
):
    if stimulus_file is not None:
        stimulus_file.to_json.return_value = stimulus_file_to_json_ret
    if sync_file is not None:
        sync_file.to_json.return_value = sync_file_to_json_ret

    stimulus_timestamps = StimulusTimestamps(
        timestamps=None,
        stimulus_file=stimulus_file,
        sync_file=sync_file
    )

    if raises:
        with pytest.raises(RuntimeError, match=raises):
            _ = stimulus_timestamps.to_json()
    else:
        obt = stimulus_timestamps.to_json()
        assert obt == expected


@pytest.mark.parametrize("behavior_session_id, ophys_experiment_id", [
    (
        12345,
        None
    ),
    (
        1234,
        5678
    )
])
def test_stimulus_timestamps_from_lims(
    monkeypatch, behavior_session_id, ophys_experiment_id
):
    mock_db_conn = create_autospec(PostgresQueryMixin, instance=True)

    mock_stimulus_file = create_autospec(StimulusFile)
    mock_sync_file = create_autospec(SyncFile)

    mock_get_behavior_stimulus_timestamps = create_autospec(
        get_behavior_stimulus_timestamps
    )
    mock_get_ophys_stimulus_timestamps = create_autospec(
        get_ophys_stimulus_timestamps
    )

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps.StimulusFile",
            mock_stimulus_file
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps.SyncFile",
            mock_sync_file
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps"
            ".get_behavior_stimulus_timestamps",
            mock_get_behavior_stimulus_timestamps
        )
        m.setattr(
            "allensdk.brain_observatory.behavior.data_objects"
            ".timestamps.stimulus_timestamps.stimulus_timestamps"
            ".get_ophys_stimulus_timestamps",
            mock_get_ophys_stimulus_timestamps
        )
        mock_stimulus_file_instance = mock_stimulus_file.from_lims(
            mock_db_conn, behavior_session_id
        )
        ts_from_stim = StimulusTimestamps.from_stimulus_file(
            stimulus_file=mock_stimulus_file_instance)
        assert ts_from_stim._stimulus_file == mock_stimulus_file_instance

        if behavior_session_id is not None and ophys_experiment_id is not None:
            mock_sync_file_instance = mock_sync_file.from_lims(
                mock_db_conn, ophys_experiment_id
            )
            ts_from_sync = StimulusTimestamps.from_sync_file(
                sync_file=mock_sync_file_instance)

    if behavior_session_id is not None and ophys_experiment_id is not None:
        mock_get_ophys_stimulus_timestamps.assert_called_once_with(
            sync_path=mock_sync_file_instance.filepath
        )
        assert ts_from_sync._sync_file == mock_sync_file_instance
    else:
        mock_stimulus_file.from_lims.assert_called_with(
            mock_db_conn, behavior_session_id
        )
        mock_get_behavior_stimulus_timestamps.assert_called_once_with(
            stimulus_pkl=mock_stimulus_file_instance.data
        )


# Fixtures:
# nwbfile:
#   test/brain_observatory/behavior/conftest.py
# data_object_roundtrip_fixture:
#   test/brain_observatory/behavior/data_objects/conftest.py
@pytest.mark.parametrize('roundtrip, stimulus_timestamps_data', [
    (True, np.array([1, 2, 3, 4, 5])),
    (True, np.array([6, 7, 8, 9, 10])),
    (False, np.array([11, 12, 13, 14, 15])),
    (False, np.array([16, 17, 18, 19, 20]))
])
def test_stimulus_timestamps_nwb_roundtrip(
    nwbfile, data_object_roundtrip_fixture, roundtrip, stimulus_timestamps_data
):
    stimulus_timestamps = StimulusTimestamps(
        timestamps=stimulus_timestamps_data
    )
    nwbfile = stimulus_timestamps.to_nwb(nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(nwbfile, StimulusTimestamps)
    else:
        obt = StimulusTimestamps.from_nwb(nwbfile)

    assert np.allclose(obt.value, stimulus_timestamps_data)
