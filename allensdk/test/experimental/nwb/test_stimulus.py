from allensdk.experimental.nwb.stimulus import (VisualBehaviorStimulusAdapter,
                                                VisualCodingStimulusAdapter)
from pynwb import NWBFile, NWBHDF5IO
import datetime
import os
import numpy as np
import pytest


@pytest.fixture
def vb_pkl():
    return '/allen/programs/braintv/production/visualbehavior/prod0/specimen_710269829/ophys_session_759332825/759332825_stim.pkl'

@pytest.fixture
def vb_sync():
    return '/allen/programs/braintv/production/visualbehavior/prod0/specimen_710269829/ophys_session_759332825/759332825_sync.h5'

@pytest.fixture
def vc_pkl():
    return '/allen/programs/braintv/production/neuralcoding/prod53/specimen_691654617/ophys_session_714254764/714254764_388801_20180626_stim.pkl'

@pytest.fixture
def vc_sync():
    return '/allen/programs/braintv/production/neuralcoding/prod53/specimen_691654617/ophys_session_714254764/714254764_388801_20180626_sync.h5'

@pytest.fixture
def nwb_filename(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test").join("test.nwb")
    return str(nwb)


def test_visual_behavior_running_speed(nwb_filename, vb_pkl, vb_sync):
    adapter = VisualBehaviorStimulusAdapter(vb_pkl, vb_sync, stim_key=2)

    nwbfile = NWBFile(
        source='Data source',
        session_description='test foraging2',
        identifier='behavior_session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    running_speed = adapter.running_speed
    nwbfile.add_acquisition(running_speed)

    with NWBHDF5IO(nwb_filename, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(nwb_filename, mode='r').read()
    running_speed_in = nwbfile_in.get_acquisition('running_speed')
    assert(np.allclose(running_speed_in.data, running_speed.data))
    assert(np.allclose(running_speed_in.timestamps, running_speed.timestamps))


def test_visual_coding_running_speed(nwb_filename, vc_pkl, vc_sync):
    adapter = VisualCodingStimulusAdapter(vc_pkl, vc_sync)

    nwbfile = NWBFile(
        source='Data source',
        session_description='test vc',
        identifier='vc_session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    running_speed = adapter.running_speed
    nwbfile.add_acquisition(running_speed)

    with NWBHDF5IO(nwb_filename, mode='w') as io:
        io.write(nwbfile)

    nwbfile_in = NWBHDF5IO(nwb_filename, mode='r').read()
    running_speed_in = nwbfile_in.get_acquisition('running_speed')
    assert(np.allclose(running_speed_in.data, running_speed.data))
    assert(np.allclose(running_speed_in.timestamps, running_speed.timestamps))
