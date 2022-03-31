import pytest
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO, NWBFile
import tempfile
import pathlib
import pytz
import datetime

from unittest.mock import patch

from allensdk.brain_observatory.ecephys.data_objects.\
    running_speed.vbn_running_speed import (
        VBNRunningSpeed)


@pytest.mark.parametrize('filtered', [True, False])
def test_vbn_running_speed_round_trip(
        behavior_pkl_fixture,
        replay_pkl_fixture,
        mapping_pkl_fixture,
        tmp_path_factory,
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

    dict_repr = {
        'sync_file': 'garbage',
        'behavior_stimulus_file': behavior_pkl_fixture['path_to_pkl'],
        'replay_stimulus_file': replay_pkl_fixture['path_to_pkl'],
        'mapping_stimulus_file': mapping_pkl_fixture['path_to_pkl']}

    def dummy_get_frame_times(sync_path=None):
        nt = behavior_pkl_fixture['n_frames']
        nt += replay_pkl_fixture['n_frames']
        nt += mapping_pkl_fixture['n_frames']
        return np.linspace(0., 10., nt)

    to_replace = ('allensdk.brain_observatory.ecephys.data_objects.'
                  'running_speed.multi_stim_running_processing.'
                  '_get_frame_times')
    with patch(to_replace, new=dummy_get_frame_times):
        running_obj = VBNRunningSpeed.from_json(
                            dict_repr=dict_repr,
                            filtered=filtered)

        new_dict = running_obj.to_json()
        new_obj = VBNRunningSpeed.from_json(
                            dict_repr=new_dict,
                            filtered=filtered)

        pd.testing.assert_frame_equal(new_obj.value, running_obj.value)

        start_time = pytz.utc.localize(datetime.datetime.fromisoformat(
                                           '2020-07-11'))
        nwbfile = NWBFile(
                       session_description='running',
                       identifier='00001',
                       session_start_time=start_time)

        running_obj.to_nwb(nwbfile=nwbfile)
        with NWBHDF5IO(nwb_path, 'w') as out_file:
            out_file.write(nwbfile)

        with NWBHDF5IO(nwb_path, 'r') as in_file:
            new_obj = VBNRunningSpeed.from_nwb(
                            nwbfile=in_file.read(),
                            filtered=filtered)

        pd.testing.assert_frame_equal(new_obj.value, running_obj.value)

    if nwb_path.exists():
        nwb_path.unlink()
