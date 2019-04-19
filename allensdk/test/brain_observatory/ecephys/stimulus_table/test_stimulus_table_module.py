import collections
import sys
import os

import pytest
import mock
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.stimulus_table.__main__ import build_stimulus_table


# psuedofixture for a stimulus pickle
def stim_file(*a, **k):
    stim_data = {
        "pre_blank_sec": 20.0,
        "frames_per_second": 60.0,
        "stimuli": [
            {
                'stim_path': 'C:\\ecephys_stimulus_scripts\\gabor_20_deg_250ms.stim',
                'display_sequence': np.array([[0, 1200]], dtype=np.int32),
                'dimnames': ['Pos', 'TF', 'SF', 'Ori', 'Contrast']
            }
        ]
    }
    StimFileClass = collections.namedtuple("StimFileClass", stim_data.keys())
    return StimFileClass(**stim_data)


@mock.patch(
    "allensdk.brain_observatory.ecephys.file_io.stim_file.CamStimOnePickleStimFile.factory", 
    new=stim_file
)
def test_build_stimulus_table(tmpdir_factory):

    tmpdir = str(tmpdir_factory.mktemp('ecephys_stimulus_table_integration'))
    table_path = os.path.join(tmpdir, 'stimulus_table.csv')
    frame_times_path = os.path.join(tmpdir, 'frame_times.npy')

    build_stimulus_table(
        stimulus_pkl_path='fake_stim_path',
        sync_h5_path='fake_sync_path',
        frame_time_strategy='use_photodiode',
        minimum_spontaneous_activity_duration=sys.float_info.epsilon,
        extract_const_params_from_repr=True,
        drop_const_params=["name", "maskParams", "win", "autoLog", "autoDraw"],
        maximum_expected_spontanous_activity_duration=99999999999,
        stimulus_name_map={
            "": "spontaneous",
            "Natural Images": "natural_scenes",
            "flash_250ms": "flash",
            "contrast_response": "drifting_gratings_contrast",
            "gabor_20_deg_250ms": "gabor",
        },
        column_name_map={},
        output_stimulus_table_path=table_path,
        output_frame_times_path=frame_times_path
    )

    obtained_table = pd.read_csv(table_path)
    obtained_frame_times = np.load(frame_times_path, allow_pickle=False)

    assert False