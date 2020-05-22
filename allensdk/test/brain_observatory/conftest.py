import pytest
import os
from datetime import datetime
import pynwb
import pandas as pd
import numpy as np


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
def stimulus_presentations():
    return pd.DataFrame({
        'alpha': [0.5, 0.4, 0.3, 0.2, 0.1],
        'start_time': [1., 2., 4., 5., 6.],
        'stimulus_name': ['gabors', 'gabors', 'random', 'movie', 'gabors'],
        'stop_time': [2., 4., 5., 6., 8.]
    }, index=pd.Index(name='stimulus_presentations_id', data=[0, 1, 2, 3, 4]))


@pytest.fixture
def roundtripper(tmpdir_factory):
    def f(nwbfile, api_cls, **api_kwargs):
        tmpdir = str(tmpdir_factory.mktemp('nwb_roundtrip_tests'))
        nwb_path = os.path.join(tmpdir, 'nwbfile.nwb')

        with pynwb.NWBHDF5IO(nwb_path, 'w') as write_io:
            write_io.write(nwbfile)

        return api_cls(nwb_path, **api_kwargs)
    return f


@pytest.fixture
def stimulus_timestamps():
    return np.array([1., 2., 3.])
