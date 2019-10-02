import pytest
import pandas as pd
import numpy as np

import allensdk.brain_observatory.ecephys.stimulus_table.ephys_pre_spikes as ephys_pre_spikes


def stimulus_psuedofixture_0():
    return {
        "display_sequence": [[500, 1000]],
        "sweep_frames": [[10, 20], [20, 25]],
        "sweep_order": [1, 0],
        "dimnames": ["a", "b", "c"],
        "sweep_table": [[1, 2, 3], [-3, -2, -1]],
        "stim_path": r"C:\\ecephys_stimulus_scripts\\gabor_20_deg_250ms.stim",
    }


def stimulus_psuedofixture_1():
    return {
        "display_sequence": [[500, 1000]],
        "sweep_frames": [[10, 20], [20, 25]],
        "sweep_order": [1, 0],
        "dimnames": [],
        "sweep_table": [],
        "stim_path": r"C:\\ecephys_stimulus_scripts\\gabor_20_deg_250ms.stim",
    }


def test_assign_sweep_values():

    stim_table = pd.DataFrame(
        [
            {
                "Start": 0,
                "End": 1,
                "orientation": np.nan,
                "color": np.nan,
                "sweep_number": 2,
            },
            {
                "Start": 1,
                "End": 2,
                "orientation": np.nan,
                "color": np.nan,
                "sweep_number": 0,
            },
            {
                "Start": 2,
                "End": 3,
                "orientation": np.nan,
                "color": np.nan,
                "sweep_number": 1,
            },
            {
                "Start": 3,
                "End": 4,
                "orientation": np.nan,
                "color": np.nan,
                "sweep_number": 3,
            },
        ]
    )

    sweep_table = pd.DataFrame(
        [
            {"orientation": 0, "color": "red", "sweep_number": 0},
            {"orientation": 45, "color": "blue", "sweep_number": 1},
            {"orientation": 90, "color": "green", "sweep_number": 2},
        ]
    )

    expected = pd.DataFrame(
        [
            {"Start": 0, "End": 1, "orientation": 90, "color": "green"},
            {"Start": 1, "End": 2, "orientation": 0, "color": "red"},
            {"Start": 2, "End": 3, "orientation": 45, "color": "blue"},
            {"Start": 3, "End": 4, "orientation": np.nan, "color": np.nan},
        ]
    )

    obtained = ephys_pre_spikes.assign_sweep_values(stim_table, sweep_table)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_column_type=False, check_dtype=False)


@pytest.mark.parametrize(
    "table,column,new_columns,drop,expected",
    [
        [
            pd.DataFrame({"Pos": [[0, 1], [1, 2]], "count": [12, 42]}),
            "Pos",
            {"Pos_x": lambda field: field[0], "Pos_y": lambda field: field[1]},
            True,
            pd.DataFrame({"Pos_x": [0, 1], "Pos_y": [1, 2], "count": [12, 42]}),
        ],
        [
            pd.DataFrame({"dog": [1, 2, 3]}),
            "cat",
            {},
            False,
            pd.DataFrame({"dog": [1, 2, 3]}),
        ],
    ],
)
def test_split_column(table, column, new_columns, drop, expected):

    obtained = ephys_pre_spikes.split_column(table, column, new_columns, drop)
    pd.testing.assert_frame_equal(expected, obtained, check_like=True, check_column_type=False, check_dtype=False)


@pytest.mark.parametrize(
    "sweeps,disp_seq,expected",
    [
        [
            {"Start": [0, 1, 2, 3], "End": [1, 2, 3, 4]},
            [[2, 5]],
            {"Start": [2, 3, 4], "End": [3, 4, 5], "stimulus_block": [0, 0, 0]},
        ],
        [
            {"Start": [1, 3, 15, 18], "End": [3, 4, 18, 20]},
            [[2, 4], [16, 36]],
            {
                "Start": [3, 17, 29, 32],
                "End": [5, 18, 32, 34],
                "stimulus_block": [0, 1, 1, 1],
            },
        ],
    ],
)
def test_apply_display_sequence(sweeps, disp_seq, expected):

    table = pd.DataFrame(sweeps)
    disp_seq = np.array(disp_seq)
    obt_table = ephys_pre_spikes.apply_display_sequence(table, disp_seq)

    expected_table = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(
        obt_table, expected_table, check_like=True, check_column_type=False, check_dtype=False
    )


@pytest.mark.parametrize(
    "stimulus_tables,expected",
    [
        [
            [
                pd.DataFrame({"Start": [0, 1], "End": [1, 2]}),
                pd.DataFrame({"Start": [5], "End": [7]}),
            ],
            [pd.DataFrame({"Start": [2], "End": [5]})],
        ],
        [[], []],
    ],
)
def test_make_spontaneous_activity_tables(stimulus_tables, expected):

    obtained = ephys_pre_spikes.make_spontaneous_activity_tables(stimulus_tables)

    if len(obtained) == 1:
        pd.testing.assert_frame_equal(obtained[0], expected[0], check_like=True, check_column_type=False, check_dtype=False)
    else:
        assert len(obtained) == len(expected)


# TODO: this test is really weird
@pytest.mark.parametrize(
    "stimuli,stim_tabler,spon_tabler,sort_key,expected",
    [
        [
            [[1, 5], [3, 4]],
            lambda stimulus: [pd.DataFrame({"parameter": stimulus})],
            lambda stimuli: [pd.DataFrame({"parameter": [len(stimuli)]})],
            "parameter",
            pd.DataFrame(
                {
                    "parameter": [1, 2, 3, 4, 5],
                    "stimulus_block": [0, None, 1, 1, 0],
                    "stimulus_index": [0, None, 1, 1, 0],
                }
            ),
        ]
    ],
)
def test_create_stim_table(stimuli, stim_tabler, spon_tabler, sort_key, expected):

    obtained = ephys_pre_spikes.create_stim_table(
        stimuli, stim_tabler, spon_tabler, sort_key
    )
    pd.testing.assert_frame_equal(
        obtained, expected, check_like=True, check_dtype=False, check_column_type=False
    )


@pytest.mark.parametrize(
    "stim_table,frame_times,fps,eft,map_cols,expected",
    [
        [
            pd.DataFrame(
                {"Start": [1, 2, 3, 4], "End": [2, 3, 4, 5], "data": [-1, -2, -3, -4]}
            ),
            np.array([100, 50, 25, 12.5, 6.25]),
            10,
            True,
            ("Start", "End"),
            pd.DataFrame(
                {
                    "Start": [50, 25, 12.5, 6.25],
                    "End": [25, 12.5, 6.25, 6.35],
                    "data": [-1, -2, -3, -4],
                }
            ),
        ]
    ],
)
def test_apply_frame_times(stim_table, frame_times, fps, eft, map_cols, expected):

    obtained = ephys_pre_spikes.apply_frame_times(
        stim_table, frame_times, fps, eft, map_cols
    )
    pd.testing.assert_frame_equal(obtained, expected, check_like=True, check_column_type=False, check_dtype=False)


@pytest.mark.parametrize(
    "stimulus,stf,start_key,end_key,expected",
    [
        [
            stimulus_psuedofixture_0(),
            lambda x: np.array(x),
            "Start",
            "End",
            [
                pd.DataFrame(
                    {
                        "Start": [510, 520],
                        "End": [521, 526],
                        "a": [-3, 1],
                        "b": [-2, 2],
                        "c": [-1, 3],
                        "stimulus_block": [0, 0],
                        "stimulus_name": ["gabor_20_deg_250ms"] * 2,
                    }
                )
            ],
        ],
        [
            stimulus_psuedofixture_1(),
            lambda x: np.array(x),
            "Start",
            "End",
            [
                pd.DataFrame(
                    {
                        "Start": [510, 520],
                        "End": [521, 526],
                        "Image": [1, 0],
                        "stimulus_block": [0, 0],
                        "stimulus_name": ["gabor_20_deg_250ms"] * 2,
                    }
                )
            ],
        ],
    ],
)
def test_build_stimuluswise_table(stimulus, stf, start_key, end_key, expected):

    obtained = ephys_pre_spikes.build_stimuluswise_table(
        stimulus, stf, start_key, end_key
    )
    for obtained_table, expected_table in zip(obtained, expected):
        pd.testing.assert_frame_equal(obtained_table, expected_table, check_like=True, check_column_type=False, check_dtype=False)
