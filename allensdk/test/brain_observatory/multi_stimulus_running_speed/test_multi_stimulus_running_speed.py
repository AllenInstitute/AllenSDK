import pytest
import os
import tempfile
import pathlib
import pandas as pd

from allensdk.brain_observatory.\
    multi_stimulus_running_speed.multi_stimulus_running_speed import (
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


@pytest.mark.requires_bamboo
@pytest.fixture(scope="session")
def sync_h5_path_fixture():
    sync_h5_path = os.path.join(
        DATA_DIR,
        '1090803859_553960_20210317.sync'
    )
    return sync_h5_path


@pytest.mark.requires_bamboo
@pytest.fixture(scope="session")
def pkl_path_fixture():
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

    return {'behavior': behavior_pkl_path,
            'mapping': mapping_pkl_path,
            'replay': replay_pkl_path}


@pytest.mark.requires_bamboo
@pytest.fixture(scope="session")
def multi_stimulus_fixture(tmpdir_factory,
                           sync_h5_path_fixture,
                           pkl_path_fixture):

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

    args = {
        'mapping_pkl_path': pkl_path_fixture['mapping'],
        'behavior_pkl_path': pkl_path_fixture['behavior'],
        'replay_pkl_path': pkl_path_fixture['replay'],
        'sync_h5_path': sync_h5_path_fixture,
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
@pytest.mark.requires_bamboo
def test_proccessing(multi_stimulus_fixture):

    multi_stimulus_fixture.process()

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
