import pytest
import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.current_source_density import _current_source_density as csd
from allensdk.brain_observatory.ecephys.current_source_density import _interpolation_utils as interp_utils
from allensdk.brain_observatory.ecephys.current_source_density import _filter_utils as filt_utils


@pytest.fixture
def stim_table():
    return pd.DataFrame({
        'Start': [0, 1, 2, 3, 4, 5, 6],
        'End': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        'alpha': [None, -1, -2, -3, -4, -5, -6],
        'stimulus_name': [None, 'a', 'a', 'a', 'b', 'b', 'a'],
        'stimulus_index': [None, 0, 0, 0, 1, 1, 2]
    })


# ------------ _current_source_density.py ------------
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
        stim_table, stim_name, time_step, pre_stim_time,
        post_stim_time, num_trials, stim_index
    )

    assert np.allclose(obtained, expected)
    assert np.allclose(obt_rel, exp_rel)


@pytest.mark.parametrize('times,raw,channels,windows,volts_per_bit,expected', [
    [
        np.arange(10),
        np.arange(50).reshape([10, 5]),
        [1, 3],
        [[5.5, 6], [7, 8]],
        1.0,
        [
            # data are rounded to int
            [[28, 31], [30, 33]],
            [[36, 41], [38, 43]]
        ]
    ],
    [
        np.arange(10),
        np.arange(50).reshape([10, 5]),
        [1, 3],
        [[5.5, 6], [7, 8]],
        0.5,
        [
            # volts_per_bit scaling may result in floats
            [[14, 15.5], [15, 16.5]],
            [[18, 20.5], [19, 21.5]]
        ]
    ]
])
def test_accumulate_lfp_data(times, raw, channels, windows, volts_per_bit,
                             expected):
    obtained = csd.accumulate_lfp_data(times, raw, channels,
                                       windows, volts_per_bit)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize('trial_mean_accumulated,spacing,expected,expected_channels', [
    [
        np.nanmean(np.arange(36).reshape([2, 6, 3]) ** 3, axis=0),
        1.0,
        [[1728.,  1926.,  2142.],
         [648.,   702.,   756.],
         [810.,   864.,   918.],
         [972.,  1026.,  1080.],
         [1134.,  1188.,  1242.],
         [-5292., -5706., -6138.]],
        np.arange(6)
    ]
])
def test_compute_csd(trial_mean_accumulated, spacing, expected, expected_channels):

    obtained, obtained_channels = csd.compute_csd(trial_mean_accumulated, spacing=spacing)

    assert np.allclose(obtained, expected)
    assert np.allclose(obtained_channels, expected_channels)


# ------------ _interpolation_utils.py ------------
@pytest.mark.parametrize('min_chan, max_chan, expected', [
    [
        # min_chan
        0,
        # max_chan
        4,
        # expected actual channel locations
        [[16, 0], [48,  0], [0,  20], [32, 20]]
    ],
    [
        2,
        6,
        [[0, 20], [32, 20], [16, 40], [48, 40]]
    ],
    [
        0,
        8,
        [[16,  0], [48,  0], [0, 20], [32, 20],
         [16, 40], [48, 40], [0, 60], [32, 60]]
    ],
    [
        4,
        8,
        [[16, 40], [48, 40], [0, 60], [32, 60]]
    ],
    [
        5,
        6,
        [[48, 40]]
    ]

])
def test_make_actual_channel_locations(min_chan, max_chan, expected):
    obtained = interp_utils.make_actual_channel_locations(min_chan=min_chan,
                                                          max_chan=max_chan)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize('min_chan, max_chan, expected', [
    [
        # min_chan
        0,
        # max_chan
        7,
        # expected interpolated channel locations
        [[24, 0], [24, 10], [24, 20], [24, 30], [24, 40], [24, 50], [24, 60]]
    ],
    [
        0,
        14,
        [[24,  0], [24, 10], [24, 20], [24,  30], [24,  40], [24,  50], [24,  60],
         [24, 70], [24, 80], [24, 90], [24, 100], [24, 110], [24, 120], [24, 130]]
    ],
    [
        2,
        6,
        [[24, 20], [24, 30], [24, 40], [24, 50]]
    ],
    [
        7,
        14,
        [[24, 70], [24, 80], [24, 90], [24, 100], [24, 110], [24, 120], [24, 130]]
    ],
    [
        8,
        9,
        [[24, 80]]
    ]
])
def test_make_interp_channel_locations(min_chan, max_chan, expected):
    obtained = interp_utils.make_interp_channel_locations(min_chan=min_chan,
                                                          max_chan=max_chan)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize('lfp, actual_locs, interp_locs, expected', [
    [
        # lfp
        np.arange(36).reshape([2, 6, 3]) ** 3,
        # actual_locs
        interp_utils.make_actual_channel_locations(0, 6),
        # interp_locs
        interp_utils.make_interp_channel_locations(0, 6),
        # expected (interp_lfp, spacing)
        ([[[-1.48688877e+01, -1.65987508e+00,  2.25788198e+01],
           [1.84977651e+02,  3.01621496e+02,  4.52968522e+02],
           [5.82039685e+02,  8.18476063e+02,  1.11005335e+03],
           [1.23712914e+03,  1.61285986e+03,  2.05929375e+03],
           [2.03821497e+03,  2.56276850e+03,  3.17035171e+03],
           [0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],

          [[6.80643377e+03,  7.93617702e+03,  9.18494994e+03],
           [1.24901530e+04,  1.41494541e+04,  1.59514583e+04],
           [1.81704531e+04,  2.03174258e+04,  2.26275394e+04],
           [2.37138688e+04,  2.62802568e+04,  2.90253479e+04],
           [2.90797202e+04,  3.20168080e+04,  3.51449255e+04],
           [0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]], 0.01)
    ]
])
def test_interp_channel_locs(lfp, actual_locs, interp_locs, expected):
    obtained = interp_utils.interp_channel_locs(lfp=lfp,
                                                actual_locs=actual_locs,
                                                interp_locs=interp_locs)

    obtained_interp_lfp, obtained_spacing = obtained
    expected_interp_lfp, expected_spacing = expected

    assert np.allclose(obtained_interp_lfp, expected_interp_lfp)
    assert obtained_spacing == expected_spacing


# ------------ _filter_utils.py ------------
@pytest.mark.parametrize('lfp, ref_channels, noisy_thresh, expected', [
    [
        # lfp arrays in the form of: trials x channel x time samples
        # channel 1 should be marked as 'noisy' and 2 should be removed
        # for being a reference
        np.array([[[0.1, 0.1, 0.1, 0.1], [0, 50, 500, 5000], [0, 0, 0, 0], [0.3, 0.3, 0.3, 0.3]],
                  [[0.15, 0.15, 0.15, 0.15], [0, 10, 100, 1000], [0, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]],
                  [[0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0], [0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2]]]),
        # reference channels
        [2],
        # noisy_channel_threshold
        2.0,
        # expected output (cleaned_lfp, good_indices)
        (np.array([[[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3]],
                  [[0.15, 0.15, 0.15, 0.15], [0.25, 0.25, 0.25, 0.25]],
                  [[0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]]]),
         np.array([0, 3]))
    ]
])
def test_select_good_channels(lfp, ref_channels, noisy_thresh, expected):
    obtained = filt_utils.select_good_channels(lfp,
                                               ref_channels,
                                               noisy_thresh)
    obtained_cleaned, obtained_good_inds = obtained
    assert np.allclose(obtained_cleaned, expected[0])
    assert np.allclose(obtained_good_inds, expected[1])


@pytest.mark.parametrize('lfp, sampling_rate, filter_cuts, filter_order, expected', [
    [
        # lfp
        np.arange(30).reshape([1, 3, 10]),
        # sampling_rate
        1000,
        # filter_cuts
        [5.0, 150.0],
        # filter_order
        1,
        # expected output
        [[[-8.03033681, -7.51102701, -7.00015769, -6.49716665, -6.00149202,
           -5.51257727, -5.02987273, -4.55283625, -4.08093445, -3.61364687],
          [-8.03033681, -7.51102701, -7.00015769, -6.49716665, -6.00149202,
           -5.51257727, -5.02987273, -4.55283625, -4.08093445, -3.61364687],
          [-8.03033681, -7.51102701, -7.00015769, -6.49716665, -6.00149202,
           -5.51257727, -5.02987273, -4.55283625, -4.08093445, -3.61364687]]]

    ]

])
def test_filter_lfp_channels(lfp, sampling_rate, filter_cuts, filter_order, expected):
    obtained = filt_utils.filter_lfp_channels(lfp,
                                              sampling_rate,
                                              filter_cuts,
                                              filter_order)
    assert np.allclose(obtained, expected)
