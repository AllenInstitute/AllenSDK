import pytest
import os
import tempfile
import pathlib
import pandas as pd

from allensdk.brain_observatory.\
    filtered_running_speed.multi_stimulus_running_speed import (
        MultiStimulusRunningSpeed
    )

DATA_DIR = os.environ.get(
    "ECEPHYS_PIPELINE_DATA",
    os.path.join(
        "/",
        "allen",
        "aibs",
        "informatics",
        "module_test_data",
        "ecephys",
        "filtered_running_speed"
    ),
)

NUMBER_OF_VSYNCS = 523198
NUMBER_OF_NET_ROTATION_ITEMS = 522974
BEHAVIOR_START = 0
MAPPING_START = 215999
REPLAY_START = 307199


@pytest.fixture(scope="session")
def multi_stimulus_fixture(tmpdir_factory):

    temp_output_dir = pathlib.Path(
        tmpdir_factory.mktemp('MultiStimulusRunningSpeedOutput')
    )

    output_path = tempfile.mkstemp(
        dir=temp_output_dir,
        prefix='output_',
        suffix='.h5')[1]

    output_json = tempfile.mkstemp(
        dir=temp_output_dir,
        prefix='output_',
        suffix='.json')[1]

    mapping_pkl_path = os.path.join(
        DATA_DIR,
        '1090803859_553960_20210317_mapping.pkl'
    )

    behavior_pkl_path = os.path.join(
        DATA_DIR,
        '1090803859_553960_20210317_behavior.pkl'
    )

    replay_pkl_path = os.path.join(
        DATA_DIR,
        '1090803859_553960_20210317_replay.pkl'
    )

    sync_h5_path = os.path.join(
        DATA_DIR,
        '1090803859_553960_20210317.sync'
    )

    args = {
        'mapping_pkl_path': mapping_pkl_path,
        'behavior_pkl_path': behavior_pkl_path,
        'replay_pkl_path': replay_pkl_path,
        'sync_h5_path': sync_h5_path,
        'output_json': output_json,
        'output_path': output_path,
        'use_lowpass_filter': True,
        'zscore_threshold': 10.0
    }

    return MultiStimulusRunningSpeed(
        args=[],
        input_data=args
    )


# smoke test
def test_proccessing(multi_stimulus_fixture):
    try:
        multi_stimulus_fixture.process()
    except Exception as exception:
        assert False, f"raised an exception {exception}"

    output_path = multi_stimulus_fixture.args['output_path']

    obtained_velocity = pd.read_hdf(
        output_path,
        key="running_speed"
    )

    obtained_raw = pd.read_hdf(
        output_path,
        key="raw_data"
    )

    # check that keys exist
    assert('net_rotation' in obtained_velocity)
    assert('velocity' in obtained_velocity)
    assert('frame_indexes' in obtained_velocity)
    assert('frame_time' in obtained_velocity)
    assert('vsig' in obtained_raw)
    assert('vin' in obtained_raw)
    assert('frame_time' in obtained_raw)
    assert('dx' in obtained_raw)

    # check that the data is the correct length
    assert(len(obtained_raw['frame_time']) == NUMBER_OF_VSYNCS)
    assert(
        len(obtained_velocity['net_rotation']) == NUMBER_OF_NET_ROTATION_ITEMS
    )


def test_get_stimulus_starts_and_ends(multi_stimulus_fixture):
    (
        behavior_start,
        mapping_start,
        replay_start,
        replay_end
    ) = multi_stimulus_fixture._get_stimulus_starts_and_ends()

    assert(behavior_start == BEHAVIOR_START)
    assert(mapping_start == MAPPING_START)
    assert(replay_start == REPLAY_START)
    assert(replay_end == NUMBER_OF_VSYNCS)


def test_get_frame_times(multi_stimulus_fixture):
    number_of_frames = len(multi_stimulus_fixture._get_frame_times())

    assert(number_of_frames == NUMBER_OF_VSYNCS)
