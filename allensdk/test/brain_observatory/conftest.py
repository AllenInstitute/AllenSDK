import pytest
import os
from datetime import datetime
import pynwb
import numpy as np

from allensdk.brain_observatory.vbn_2022.input_json_writer.utils import (
    vbn_nwb_config_from_ecephys_session_id_list)


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


@pytest.fixture
def data_object_roundtrip_fixture(tmp_path):
    def f(nwbfile, data_object_cls, **data_object_cls_kwargs):
        tmp_dir = tmp_path / "data_object_nwb_roundtrip_tests"
        tmp_dir.mkdir()
        nwb_path = tmp_dir / "data_object_roundtrip_nwbfile.nwb"

        with pynwb.NWBHDF5IO(str(nwb_path), 'w') as write_io:
            write_io.write(nwbfile)

        with pynwb.NWBHDF5IO(str(nwb_path), 'r') as read_io:
            roundtripped_nwbfile = read_io.read()

            data_object_instance = data_object_cls.from_nwb(
                roundtripped_nwbfile, **data_object_cls_kwargs
            )

        return data_object_instance

    return f


@pytest.fixture(scope='session')
def behavior_ecephys_session_config_fixture():
    """
    Return a dict representing the session_data needed to create
    a BehaviorEcephysSession
    """
    session_data_list = vbn_nwb_config_from_ecephys_session_id_list(
        ecephys_session_id_list=[1111216934],
        probes_to_skip=None)

    return session_data_list['sessions'][0]
