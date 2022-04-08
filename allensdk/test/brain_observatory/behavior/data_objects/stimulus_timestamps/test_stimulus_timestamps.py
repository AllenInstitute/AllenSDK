from pathlib import Path

import pytest
from unittest.mock import create_autospec

import numpy as np

from itertools import product

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import (
    BehaviorStimulusFile, SyncFile
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
    mock_stimulus_file = create_autospec(BehaviorStimulusFile)
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
            ".timestamps.stimulus_timestamps"
            ".stimulus_timestamps.BehaviorStimulusFile",
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
            stimulus_file=mock_stimulus_file_instance,
            monitor_delay=0.0)

        if has_pkl and has_sync:
            mock_sync_file_instance = mock_sync_file.from_json(dict_repr)
            ts_from_sync = StimulusTimestamps.from_sync_file(
                sync_file=mock_sync_file_instance,
                monitor_delay=0.0)

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


@pytest.fixture
def stimulus_file_fixture():
    dir = Path(__file__).parent.parent.resolve()
    test_data_dir = dir / 'test_data'
    sf_path = test_data_dir / 'stimulus_file.pkl'

    return BehaviorStimulusFile.from_json(
        dict_repr={'behavior_stimulus_file': str(sf_path)})


def test_stimulus_timestamps_from_json2(stimulus_file_fixture):

    sf = stimulus_file_fixture
    stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
        stimulus_file=sf,
        monitor_delay=0.0)
    expected = np.array([0.016 * i for i in range(11)])
    assert np.allclose(expected, stimulus_timestamps.value)


@pytest.mark.parametrize("delay", [0.0, 0.5])
def test_stimulus_timestamps_json_roundtrip(stimulus_file_fixture, delay):

    sf = stimulus_file_fixture
    stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
        stimulus_file=sf,
        monitor_delay=delay)

    dict_repr = stimulus_timestamps.to_json()
    stimulus_timestamps_from_json = StimulusTimestamps.from_json(dict_repr)
    assert np.allclose(
        stimulus_timestamps.value,
        stimulus_timestamps_from_json.value
    )


def test_stimulus_timestamps_from_json3(stimulus_file_fixture):
    """
    Test that StimulusTimestamps.from_stimulus_file
    just returns the sum of the intervalsms field in the
    behavior stimulus pickle file, padded with a zero at the
    first timestamp.
    """

    sf = stimulus_file_fixture
    sf._data['items']['behavior']['intervalsms'] = [0.1, 0.2, 0.3, 0.4]

    stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
        stimulus_file=sf,
        monitor_delay=0.0)

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
            create_autospec(BehaviorStimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # sync_file
            create_autospec(SyncFile, instance=True),
            # sync_file_to_json_ret
            {"sync_file": "sync.h5"},
            # raises
            False,
            # expected
            {"behavior_stimulus_file": "stim.pkl", "sync_file": "sync.h5",
             "monitor_delay": 0.0}
        ),
        # Test to_json with only stimulus_file
        (
            # stimulus_file
            create_autospec(BehaviorStimulusFile, instance=True),
            # stimulus_file_to_json_ret
            {"behavior_stimulus_file": "stim.pkl"},
            # sync_file
            None,
            # sync_file_to_json_ret
            None,
            # raises
            False,
            # expected
            {"behavior_stimulus_file": "stim.pkl",
             "monitor_delay": 0.0}
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
        timestamps=np.array([0.0]),
        monitor_delay=0.0,
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

    mock_stimulus_file = create_autospec(BehaviorStimulusFile)
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
            ".timestamps.stimulus_timestamps"
            ".stimulus_timestamps.BehaviorStimulusFile",
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
            stimulus_file=mock_stimulus_file_instance,
            monitor_delay=0.0)
        assert ts_from_stim._stimulus_file == mock_stimulus_file_instance

        if behavior_session_id is not None and ophys_experiment_id is not None:
            mock_sync_file_instance = mock_sync_file.from_lims(
                mock_db_conn, ophys_experiment_id
            )
            ts_from_sync = StimulusTimestamps.from_sync_file(
                sync_file=mock_sync_file_instance,
                monitor_delay=0.0)

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
@pytest.mark.parametrize(
    'roundtrip, raw_stimulus_timestamps_data, monitor_delay',
    product((True, False),
            (np.arange(1, 6, 1), np.arange(6, 11, 1)),
            (0.1, 0.2)))
def test_stimulus_timestamps_nwb_roundtrip(
    nwbfile,
    data_object_roundtrip_fixture,
    roundtrip,
    raw_stimulus_timestamps_data,
    monitor_delay
):
    stimulus_timestamps = StimulusTimestamps(
        timestamps=raw_stimulus_timestamps_data,
        monitor_delay=monitor_delay
    )
    nwbfile = stimulus_timestamps.to_nwb(nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(nwbfile, StimulusTimestamps)
    else:
        obt = StimulusTimestamps.from_nwb(nwbfile)

    assert np.allclose(obt.value,
                       raw_stimulus_timestamps_data+monitor_delay)


def test_stimulus_timestamps_from_nwb_to_json(
    nwbfile
):
    """
    Make sure that, if a StimulusTimestamps is created from_nwb,
    it cannot be written to_json.

    When writing StimulusTimestamps to_nwb, monitor delay is already
    folded into the timestamp values so that from_nwb reads the
    timestamps in and sets monitor_delay=0.0. from_json and to_json
    depend on storing the BehaviorStimulusFile and a non-zero monitor
    delay. If we ever decide to make it possible to read a
    StimulusTimetamps from_nwb and then write it to_json, we will
    need to start writing the monitor_delay to the NWB file in a
    consistent manner.
    """
    stimulus_timestamps = StimulusTimestamps(
        timestamps=np.arange(1, 10, 1),
        monitor_delay=0.1
    )
    nwbfile = stimulus_timestamps.to_nwb(nwbfile)
    obt = StimulusTimestamps.from_nwb(nwbfile)
    with pytest.raises(RuntimeError,
                       match="information about the BehaviorStimulusFile"):
        obt.to_json()
