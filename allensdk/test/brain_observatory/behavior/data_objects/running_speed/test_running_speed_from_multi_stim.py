import pytest
import pandas as pd
from pynwb import NWBHDF5IO, NWBFile
import tempfile
import pathlib
import pytz
import datetime

from allensdk.brain_observatory.behavior.data_objects.\
    running_speed.running_speed import (
        RunningSpeed)

from allensdk.brain_observatory.behavior.data_objects.\
    running_speed.running_acquisition import (
        RunningAcquisition)


@pytest.mark.parametrize('filtered', [True, False])
def test_vbn_running_speed_round_trip(
        behavior_stim_file_fixture,
        replay_stim_file_fixture,
        mapping_stim_file_fixture,
        sync_file_fixture,
        tmp_path_factory,
        helper_functions,
        filtered):
    """
    Test that we can round trip VBNRunningSpeed between from_json, to_json,
    from_nwb, and to_nwb
    """
    tmpdir = tmp_path_factory.mktemp('vbn_running_roundtrip')
    nwb_path = pathlib.Path(
                tempfile.mkstemp(
                    dir=tmpdir,
                    suffix='.nwb')[1])

    running_obj = RunningSpeed.from_multiple_stimulus_files(
                        behavior_stimulus_file=behavior_stim_file_fixture,
                        mapping_stimulus_file=mapping_stim_file_fixture,
                        replay_stimulus_file=replay_stim_file_fixture,
                        sync_file=sync_file_fixture,
                        filtered=filtered,
                        zscore_threshold=10.0)

    start_time = pytz.utc.localize(datetime.datetime(2020, 7, 11))
    nwbfile = NWBFile(
                   session_description='running',
                   identifier='00001',
                   session_start_time=start_time)

    running_obj.to_nwb(nwbfile=nwbfile)
    with NWBHDF5IO(nwb_path, 'w') as out_file:
        out_file.write(nwbfile)

    with NWBHDF5IO(nwb_path, 'r') as in_file:
        new_obj = RunningSpeed.from_nwb(
                        nwbfile=in_file.read(),
                        filtered=filtered)

    pd.testing.assert_frame_equal(new_obj.value, running_obj.value)
    helper_functions.windows_safe_cleanup(file_path=nwb_path)


def test_vbn_running_acq_round_trip(
        behavior_stim_file_fixture,
        replay_stim_file_fixture,
        mapping_stim_file_fixture,
        sync_file_fixture,
        helper_functions,
        tmp_path_factory):
    """
    Test that we can round trip VBNRunningAcquistion
    between from_json, to_json, from_nwb, and to_nwb
    """
    tmpdir = tmp_path_factory.mktemp('vbn_running_acq_roundtrip')
    nwb_path = pathlib.Path(
                tempfile.mkstemp(
                    dir=tmpdir,
                    suffix='.nwb')[1])

    running_obj = RunningAcquisition.from_multiple_stimulus_files(
                    behavior_stimulus_file=behavior_stim_file_fixture,
                    mapping_stimulus_file=mapping_stim_file_fixture,
                    replay_stimulus_file=replay_stim_file_fixture,
                    sync_file=sync_file_fixture)

    start_time = pytz.utc.localize(datetime.datetime(202, 7, 11))
    nwbfile = NWBFile(
                   session_description='running',
                   identifier='00001',
                   session_start_time=start_time)

    running_obj.to_nwb(nwbfile=nwbfile)
    with NWBHDF5IO(nwb_path, 'w') as out_file:
        out_file.write(nwbfile)

    with NWBHDF5IO(nwb_path, 'r') as in_file:
        new_obj = RunningAcquisition.from_nwb(
                        nwbfile=in_file.read())

    pd.testing.assert_frame_equal(new_obj.value, running_obj.value)

    helper_functions.windows_safe_cleanup(file_path=nwb_path)
