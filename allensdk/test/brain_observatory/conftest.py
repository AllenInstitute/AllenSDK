import pytest
import os
from datetime import datetime
import pynwb


@pytest.fixture
def running_speed():
    from allensdk.brain_observatory.running_speed import RunningSpeed
    return RunningSpeed(
        timestamps=[1., 2., 3.],
        values=[4, 5, 6]
    )

@pytest.fixture
def nwbfile():
    return pynwb.NWBFile(
        session_description='asession',
        identifier='afile',
        session_start_time=datetime.now()
    )


@pytest.fixture
def roundtripper(tmpdir_factory):
    def f(nwbfile, api):
        tmpdir = str(tmpdir_factory.mktemp('nwb_roundtrip_tests'))
        nwb_path = os.path.join(tmpdir, 'nwbfile.nwb')

        with pynwb.NWBHDF5IO(nwb_path, 'w') as write_io:
            write_io.write(nwbfile)

        return api(nwb_path)
    return f