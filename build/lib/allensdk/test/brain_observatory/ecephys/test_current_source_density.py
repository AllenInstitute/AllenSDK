import pytest
import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.current_source_density import _current_source_density as csd


@pytest.fixture
def stim_table():
    return pd.DataFrame({
        'Start': [0, 1, 2, 3, 4, 5, 6],
        'End': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        'alpha': [None, -1, -2, -3, -4, -5, -6],
        'stimulus_name': [None, 'a', 'a', 'a', 'b', 'b', 'a'],
        'stimulus_index': [None, 0, 0, 0, 1, 1, 2]
    })


@pytest.mark.parametrize('stim_index', [0, None])
def test_extract_trial_windows(stim_table, stim_index):

    stim_name = 'a'
    time_step = 0.1
    pre_stim_time = 0.2
    post_stim_time = 0.3
    num_trials = 2

    expected = [
        [0.8, 0.9, 1.0, 1.1, 1.2],
        [1.8, 1.9, 2.0, 2.1, 2.2]
    ]
    exp_rel = [-0.2, -0.1, 0.0, 0.1, 0.2]

    obtained, obt_rel = csd.extract_trial_windows(
        stim_table, stim_name, time_step, pre_stim_time, post_stim_time, num_trials, stim_index
    )

    assert np.allclose(obtained, expected)
    assert np.allclose(obt_rel, exp_rel)


@pytest.mark.parametrize('surface,unusable,step,expected', [
    [10, [6, 8], 2, [0, 2, 4, 7, 9]],
    [10, [6, 7], 2, [0, 2, 4, 8]],
    [10, [], 4, [0, 4, 8]]
])
def test_identify_lfp_channels(surface, unusable, step, expected):

    obtained = csd.identify_lfp_channels(surface, unusable, step)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize('lfp_channels,step,expected', [
    [
        [2, 4, 7, 12],
        2,
        {
            8: {'zone': 4, 'base_zones': [3, 6], 'base_channels': [7, 12]}, 
            10: {'zone': 5, 'base_zones': [3, 6], 'base_channels': [7, 12]},
        }
    ]
])
def test_get_missing_channels(lfp_channels, step, expected):

    obtained = csd.get_missing_channels(lfp_channels, step=step)
    for channel, obt in obtained.items():
        exp = expected[channel]
        for key, value in exp.items():
            assert np.allclose(obt[key], value)


@pytest.mark.parametrize('times,raw,channels,windows,expected', [
    [ 
        np.arange(10), 
        np.arange(50).reshape([10, 5]), 
        [1, 3], 
        [[5.5, 6], [7, 8]], 
        [ 
            [ [28, 31], [30, 33] ], # data are rounded to int
            [ [36, 41], [38, 43] ] 
        ] 
    ]
])
def test_accumulate_lfp_data(times, raw, channels, windows, expected):

    obtained = csd.accumulate_lfp_data(times, raw, channels, windows)
    assert np.allclose(obtained, expected)



@pytest.mark.parametrize('accumulated,spacing,expected,expected_channels', [
    [
        np.arange(36).reshape([2, 6, 3]) ** 3,
        1.0,
        [[-54, 0, 54], [-54, 0, 54]],
        np.arange(6)
    ]
])
def test_compute_csd(accumulated, spacing, expected, expected_channels):

    obtained, obtained_channels = csd.compute_csd(accumulated, spacing=spacing)
    assert np.allclose(obtained, expected)
    assert np.allclose(obtained_channels, expected_channels)


@pytest.mark.parametrize('lfp,channels,missing_channels,exp,exp_chan', [
    [
        np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]),
        [0, 2, 7],
        {4: {'base_channels': [2, 7], 'base_zones': [1, 3], 'zone': 2}},
        np.array([ [1, 2, 3], [4, 5, 6], [5.5, 6.5, 7.5], [7, 8, 9] ]),
        [0, 2, 4, 7]
    ]
])
def test_resample_to_regular(lfp, channels, missing_channels, exp, exp_chan):

    obt, obt_chan = csd.resample_to_regular(lfp, channels, missing_channels)
    assert np.allclose(obt, exp)
    assert np.allclose(obt_chan, exp_chan)