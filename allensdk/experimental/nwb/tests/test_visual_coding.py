import datetime
import os
from allensdk.experimental.nwb.visual_coding import VisualCodingLegacyNwbAdapter
from pynwb import NWBFile, NWBHDF5IO
import numpy as np
import pytest


@pytest.fixture
def vc_nwb():
    return '/allen/programs/braintv/production/neuralcoding/prod53/specimen_691654617/ophys_session_714254764/ophys_experiment_714778446/714778446.nwb'


@pytest.fixture
def nwb_filename(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test").join("test.nwb")
    return str(nwb)


@pytest.mark.skipif(not os.environ.get('ALLENSDK_EXPERIMENTAL',''), reason='Experimental')
@pytest.mark.parametrize("compress", (True, False))
def test_legacy_cv_running_speed(nwb_filename, vc_nwb, compress):
    adapter = VisualCodingLegacyNwbAdapter(vc_nwb, compress=compress)

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
    assert(np.allclose(running_speed_in.data, running_speed.data.data, equal_nan=True))
    assert(np.allclose(running_speed_in.timestamps, running_speed.timestamps))
