from functools import partial

import pytest
import numpy as np

from allensdk.brain_observatory.ecephys import stimulus_sync


# manual test cases for compute_frame_times, allocate_by_vsync, assign_to_last
@pytest.mark.parametrize('photodiode_times,frame_duration,num_frames,cycle,vsyncs,expected', [
    [ # super basic, no vsyncs, no bad frames
        np.linspace(5, 30.0, 11), 0.25, 100, 10, None,
        [
            np.arange(5, 30, 0.25),
            np.arange(5.25, 30.25, 0.25)
        ]
    ],
    [ # also no bad frames
        np.array([5, 5.75, 6.5, 7.25]), 0.25, 9, 3, None,
        [
            np.array([5, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0]),
            np.array([5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25  ]),
        ]
    ],
    [ # now add in a long-short, using the append_to_last rule
        np.array([5, 5.75, 6.75, 7.25, 8.0]), 0.25, 12, 3, None,
        [
            np.array([5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.75, 7.00, 7.25, 7.25, 7.50, 7.75]),
            np.array([5.25, 5.50, 5.75, 6.00, 6.25, 6.75, 7.00, 7.25, 7.25, 7.50, 7.75, 8.00]),
        ]
    ],
    [ # expected timing, using vsyncs
        np.array([5, 5.75, 6.5, 7.25, 8.0]), 0.25, 12, 3,
        np.array([4.9 , 5.15, 5.4 , 5.65, 5.9 , 6.15, 6.4 , 6.65, 6.9 , 7.15, 7.4 , 7.65, 7.9]),
        [
            np.array([5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.0, 7.25, 7.50, 7.75]),
            np.array([5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.0, 7.25, 7.50, 7.75, 8.00])
        ]
    ],
    [ # classic extra frame case
        np.array([5, 5.75, 6.5, 7.5, 8.25]), 0.25, 12, 3,
        np.array([4.9 , 5.15, 5.4 , 5.65, 5.9 , 6.15, 6.4 , 6.65, 7.15, 7.4 , 7.65, 7.9 , 8.15]),
        [
            np.array([5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.25, 7.50, 7.75, 8.00]),
            np.array([5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.25, 7.50, 7.75, 8., 8.25])
        ]
    ],
    [ # long-short, using vsyncs
        np.array([5, 5.75, 6.5, 7.50, 8.0]), 0.25, 12, 3,
        np.array([4.9 , 5.15, 5.4 , 5.65, 5.9 , 6.15, 6.4 , 6.9 , 7.15, 7.4, 7.4, 7.65, 7.9]),
        [
            np.array([5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 7.0, 7.25, 7.50, 7.50, 7.75]),
            np.array([5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 7.0, 7.25, 7.50, 7.50, 7.75, 8.00])
        ]
    ],
    [ # only short, using vsyncs
        np.array([5, 5.75, 6.5, 7.0, 7.75]), 0.25, 12, 3,
        np.array([4.9 , 5.15, 5.4 , 5.65, 5.9 , 6.15, 6.4, 6.65, 6.65, 6.9, 7.15, 7.4 , 7.65, 7.9]),
        [
            np.array([5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 6.75, 7.0, 7.25, 7.50]),
            np.array([5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 6.75, 7.0, 7.25, 7.50, 7.75])
        ]
    ],
])
def test_compute_frame_times(photodiode_times, frame_duration, num_frames, cycle, vsyncs, expected):
    
    if vsyncs is not None:
        cb = partial(stimulus_sync.allocate_by_vsync, np.diff(vsyncs))
    else:
        cb = stimulus_sync.assign_to_last

    obt_indices, obt_starts, obt_ends = stimulus_sync.compute_frame_times(photodiode_times, frame_duration, num_frames, cycle, cb)
    assert(np.allclose(obt_indices, np.arange(num_frames)))
    assert np.allclose(obt_starts, expected[0])
    assert np.allclose(obt_ends, expected[1])


@pytest.mark.parametrize('process,pctiles', [
    [partial(np.random.rand, 1000), (5, 95)],
    [partial(np.random.rand, 1000), (45, 55)]
])
def test_trimmed_stats(process, pctiles):

    data = np.sort(process())
    true_mean = np.mean(data)
    true_std = np.std(data)

    lower_missing =  pctiles[0]
    upper_missing = 100 - pctiles[1]
    total_missing = lower_missing + upper_missing
    fraction_lower = lower_missing / total_missing

    num_missing = (data.size * total_missing / 100) / ( 1 - total_missing / 100 )
    num_missing_lower = int(np.around( num_missing * fraction_lower ))
    num_missing_upper = int(np.around( num_missing * ( 1 - fraction_lower ) ))

    data = np.concatenate([
        data, 
        np.zeros(num_missing_lower) - 1000,
        np.zeros(num_missing_upper) + 1000   
    ])

    obt_mean, obt_std = stimulus_sync.trimmed_stats(data, pctiles=pctiles)
    assert obt_mean == true_mean
    assert obt_std == true_std


@pytest.mark.parametrize('pd_times,vs_times, expected', [
    [ [1, 2, 3, 4, 5], [1.8, 3, 4], [2, 3, 4] ]
])
def test_trim_border_pulses(pd_times, vs_times, expected):
    obtained = stimulus_sync.trim_border_pulses(pd_times, vs_times)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize('base,effect', [
    [ np.arange(20, dtype=float), [0.25, -0.25] ],
    [ np.arange(20, dtype=float), [0.25, -0.25] ],
    # [ np.arange(20, dtype=float), [0.25, -0.4] ], misses for assymmetric cases
    # [ np.arange(20, dtype=float), [0.4, -0.25] ]
])
def test_correct_on_off_effects(base, effect):
    impacted = base.copy()
    impacted[::2] += effect[0]
    impacted[1::2] += effect[1]


    obtained = stimulus_sync.correct_on_off_effects(impacted)
    assert np.allclose(base, obtained)


@pytest.mark.parametrize('pd_times,ndevs,expected_mask', [
    [ [1, 2, 3, 9, 10, 11, 12], 4, [1, 1, 0, 0, 1, 1, 1] ],
    [ [1.03, 2.10, 2.99, 8.9, 10.0, 11.1, 11.98], 10, [1, 1, 0, 0, 1, 1, 1] ]
])
def test_flag_unexpected_edges(pd_times, ndevs, expected_mask):

    obtained_mask = stimulus_sync.flag_unexpected_edges(pd_times, ndevs)
    assert np.allclose(obtained_mask, expected_mask)


@pytest.mark.parametrize('pd_times,ndevs,cycle,max_offset,expected', [
    [ [0, 1, 2, 3, 4, 9, 10, 11], 10, 60, 5, np.arange(12) ],
    [ 
        np.concatenate([[0, 1, 2, 3, 3.95, 4, 4.1, 4.7, 9, 10, 11], np.arange(12, 1000)]), 
        0.1, 60, 5, 
        np.arange(1000) 
    ]
]) 
def test_fix_unexpected_edges(pd_times, ndevs, cycle, max_offset, expected):
    obtained = stimulus_sync.fix_unexpected_edges(pd_times, ndevs, cycle, max_offset)
    assert np.allclose(obtained, expected)


@pytest.mark.parametrize('pd_times,cycle,expected', [
    [ [0, 1, 2, 3, 4, 5.1, 6, 7, 8], 1, 1]
])
def test_estimate_frame_duration(pd_times, cycle, expected):
    obtained = stimulus_sync.estimate_frame_duration(pd_times, cycle)
    assert obtained == expected


@pytest.mark.parametrize('ends,frame_duration,irregularity,expected', [
    [ np.arange(20, dtype=float), 0.5, 1, np.concatenate([np.arange(19), [19.5]]) ]
])
def test_assign_to_last(ends, frame_duration, irregularity, expected):
    _, obt_ends = stimulus_sync.assign_to_last(None, None, ends, frame_duration, irregularity, None)
    assert np.allclose(obt_ends, expected)


@pytest.mark.parametrize('vs_diff,index,starts,ends,frame_duration,irregularity,cycle,expected', [
    [ [1, 1, 1, 1, 2, 1, 1, 1, 1], 1, [5, 6, 7], [6, 7, 8], 1, 1, 3, [[5, 6, 8], [6, 8, 9]] ],
    [ [1, 1, 1, 1, 0.5, 1, 1, 1, 1], 1, [5, 6, 7], [6, 7, 8], 1, -1, 3, [[5, 6, 6], [6, 6, 7]] ],
])
def test_allocate_by_vsync(vs_diff, index, starts, ends, frame_duration, irregularity, cycle, expected):
    obt_starts, obt_ends = stimulus_sync.allocate_by_vsync(vs_diff, index, starts, ends, frame_duration, irregularity, cycle)
    assert np.allclose(obt_starts, expected[0])
    assert np.allclose(obt_ends, expected[1])