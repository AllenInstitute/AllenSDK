import collections
import sys
import os

import pytest
import mock
import pandas as pd
import numpy as np

from allensdk.brain_observatory.ecephys.stimulus_table.__main__ import (
    build_stimulus_table,
)


def build_psuedofixture(name, data):
    new_class = collections.namedtuple(name, data.keys())
    return new_class(**data)


def stim_file(*a, **k):
    return build_psuedofixture(
        "StimFileClass",
        {
            "pre_blank_sec": 20.0,
            "frames_per_second": 10.0,
            "stimuli": [
                {
                    "stim_path": "C:\\ecephys_stimulus_scripts\\gabor_20_deg_250ms.stim",
                    "display_sequence": np.array([[50, 100]], dtype=np.int32),
                    "dimnames": ["TF", "SF"],
                    "sweep_frames": [
                        (0, 4),
                        (5, 9),
                        (10, 14),
                        (15, 19),
                        (20, 24),
                        (25, 29),
                        (30, 34),
                        (35, 39),
                    ],
                    "sweep_order": [4, 3, 7, 1, 2, 0, 5, 6],
                    "sweep_table": [
                        (-1, 1),
                        (-2, 2),
                        (-3, 3),
                        (-4, 4),
                        (-5, 5),
                        (-6, 6),
                        (-7, 7),
                        (-8, 8),
                    ],
                    "stim": "GratingStim(autoDraw=False, autoLog=True, contrast=0.8, win=Window(...))",
                },
                {
                    "stim_path": "C:\\ecephys_stimulus_scripts\\static_gratings.stim",
                    "display_sequence": np.array(
                        [[45, 46], [100, 110]], dtype=np.int32
                    ),
                    "dimnames": ["Ori", "Phase"],
                    "sweep_frames": [(0, 8), (9, 17), (18, 26), (27, 35)],
                    "sweep_order": [0, 2, 3, 1],
                    "sweep_table": [(-1.5, 1.5), (-2.5, 2.5), (-3.5, 3.5), (-4.5, 4.5)],
                    "stim": "GratingStim(contrast=0.8, ori=30.0, phase=array([0., 0.]), sf=array([0.16, 0.16]), size=array([250., 250.]))",
                },
            ],
        },
    )


def sync_file(*a, **k):
    class SyncFileClass:
        def extract_frame_times(*a, **k):
            return np.arange(10000, dtype=float) / 10

        @classmethod
        def factory(cls, *a, **k):
            return cls()

    return SyncFileClass.factory


@pytest.fixture
def expected_table():
    return pd.DataFrame(
        {
            "Start": {
                0: 0.0,
                1: 65.0,
                2: 65.9,
                3: 66.8,
                4: 70.0,
                5: 70.5,
                6: 71.0,
                7: 71.5,
                8: 72.0,
                9: 72.5,
                10: 73.0,
                11: 73.5,
                12: 74.0,
                13: 120.8,
                14: 121.7,
            },
            "End": {
                0: 65.0,
                1: 65.9,
                2: 66.8,
                3: 70.0,
                4: 70.5,
                5: 71.0,
                6: 71.5,
                7: 72.0,
                8: 72.5,
                9: 73.0,
                10: 73.5,
                11: 74.0,
                12: 120.8,
                13: 121.7,
                14: 122.6,
            },
            "stimulus_name": {
                0: "spontaneous",
                1: "static_gratings",
                2: "static_gratings",
                3: "spontaneous",
                4: "gabor",
                5: "gabor",
                6: "gabor",
                7: "gabor",
                8: "gabor",
                9: "gabor",
                10: "gabor",
                11: "gabor",
                12: "spontaneous",
                13: "static_gratings",
                14: "static_gratings",
            },
            "stimulus_block": {
                0: np.nan,
                1: 0.0,
                2: 0.0,
                3: np.nan,
                4: 1.0,
                5: 1.0,
                6: 1.0,
                7: 1.0,
                8: 1.0,
                9: 1.0,
                10: 1.0,
                11: 1.0,
                12: np.nan,
                13: 2.0,
                14: 2.0,
            },
            "Ori": {
                0: np.nan,
                1: -1.5,
                2: -3.5,
                3: np.nan,
                4: np.nan,
                5: np.nan,
                6: np.nan,
                7: np.nan,
                8: np.nan,
                9: np.nan,
                10: np.nan,
                11: np.nan,
                12: np.nan,
                13: -4.5,
                14: -2.5,
            },
            "Phase": {
                0: np.nan,
                1: 1.5,
                2: 3.5,
                3: np.nan,
                4: np.nan,
                5: np.nan,
                6: np.nan,
                7: np.nan,
                8: np.nan,
                9: np.nan,
                10: np.nan,
                11: np.nan,
                12: np.nan,
                13: 4.5,
                14: 2.5,
            },
            "contrast": {
                0: np.nan,
                1: 0.8,
                2: 0.8,
                3: np.nan,
                4: 0.8,
                5: 0.8,
                6: 0.8,
                7: 0.8,
                8: 0.8,
                9: 0.8,
                10: 0.8,
                11: 0.8,
                12: np.nan,
                13: 0.8,
                14: 0.8,
            },
            "sf": {
                0: np.nan,
                1: "[0.16, 0.16]",
                2: "[0.16, 0.16]",
                3: np.nan,
                4: "5.0",
                5: "4.0",
                6: "8.0",
                7: "2.0",
                8: "3.0",
                9: "1.0",
                10: "6.0",
                11: "7.0",
                12: np.nan,
                13: "[0.16, 0.16]",
                14: "[0.16, 0.16]",
            },
            "size": {
                0: np.nan,
                1: "[250.0, 250.0]",
                2: "[250.0, 250.0]",
                3: np.nan,
                4: np.nan,
                5: np.nan,
                6: np.nan,
                7: np.nan,
                8: np.nan,
                9: np.nan,
                10: np.nan,
                11: np.nan,
                12: np.nan,
                13: "[250.0, 250.0]",
                14: "[250.0, 250.0]",
            },
            "stimulus_index": {
                0: np.nan,
                1: 1.0,
                2: 1.0,
                3: np.nan,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
                8: 0.0,
                9: 0.0,
                10: 0.0,
                11: 0.0,
                12: np.nan,
                13: 1.0,
                14: 1.0,
            },
            "TF": {
                0: np.nan,
                1: np.nan,
                2: np.nan,
                3: np.nan,
                4: -5.0,
                5: -4.0,
                6: -8.0,
                7: -2.0,
                8: -3.0,
                9: -1.0,
                10: -6.0,
                11: -7.0,
                12: np.nan,
                13: np.nan,
                14: np.nan,
            },
        }
    )


@mock.patch(
    "allensdk.brain_observatory.ecephys.file_io.stim_file.CamStimOnePickleStimFile.factory",
    new=stim_file,
)
@mock.patch(
    "allensdk.brain_observatory.ecephys.file_io.ecephys_sync_dataset.EcephysSyncDataset.factory",
    new=sync_file(),
)
def test_build_stimulus_table(tmpdir_factory, expected_table):

    tmpdir = str(tmpdir_factory.mktemp("ecephys_stimulus_table_integration"))
    table_path = os.path.join(tmpdir, "stimulus_table.csv")
    frame_times_path = os.path.join(tmpdir, "frame_times.npy")

    build_stimulus_table(
        stimulus_pkl_path="fake_stim_path",
        sync_h5_path="fake_sync_path",
        frame_time_strategy="use_photodiode",
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
        output_frame_times_path=frame_times_path,
        fail_on_negative_duration=True
    )

    obtained_table = pd.read_csv(table_path)
    obtained_frame_times = np.load(frame_times_path, allow_pickle=False)

    pd.testing.assert_frame_equal(expected_table, obtained_table, check_like=True, check_dtype=False)
    assert np.array_equal(np.arange(10000) / 10, obtained_frame_times)
