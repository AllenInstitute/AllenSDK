import pytest
import numpy as np
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset
from allensdk.brain_observatory.sync_stim_aligner import (
    _get_start_frames,
    get_stim_timestamps_from_stimulus_blocks)


class DummyStim(object):
    """
    A class that implements num_frames;
    used for testing APIs that require
    _StimulusFile objects.
    """

    def __init__(self, n_frames):
        self._n_frames = n_frames

    @property
    def num_frames(self):
        return self._n_frames


@pytest.fixture
def line_name_fixture():
    return ['lineA', 'stim_running', 'lineB', 'vsync_stim']


@pytest.fixture
def line_to_edges_fixture(
        sync_sample_fixture):
    n_samples = len(sync_sample_fixture)

    result = dict()

    # fill lineA and lineB with random bits
    rng = np.random.default_rng(66123)
    indexes = np.arange(0, n_samples, 3)
    for line_name in ('lineA', 'lineB'):
        changes = rng.choice(indexes, 50, replace=False)
        changes = np.sort(changes)
        this_rising = []
        this_falling = []
        for idx in range(0, len(changes), 2):
            this_rising.append(changes[idx])
            this_falling.append(changes[idx+1])
        result[line_name] = {'rising_idx': np.array(this_rising),
                             'falling_idx': np.array(this_falling)}

    # create four intentional blocks of length
    # 33, 66, 99, 132 in the stim_running line
    this_rising = [12, 100, 204, 500]
    this_falling = [45, 166, 303, 632]
    result['stim_running'] = {'rising_idx': np.array(this_rising),
                              'falling_idx': np.array(this_falling)}

    # set vsync_stim lines;
    # because we are placing an edge every three frames,
    # the number of frames in each stimulus block should
    # be 1/3 that specified in the above block
    v_rising = np.arange(4, n_samples, 3, dtype=int)
    v_falling = v_rising + 1
    result['vsync_stim'] = {'rising_idx': v_rising,
                            'falling_idx': v_falling}

    return result


@pytest.fixture
def expected_start_frames_fixture(
        line_to_edges_fixture):
    """
    Return dict that maps 'rising', 'falling' to the expected
    start frames for all of the stimulus blocks in our test sync file
    """
    result = dict()
    for edge_type in ('rising', 'falling'):
        frame_edges = line_to_edges_fixture['vsync_stim'][f'{edge_type}_idx']
        stim_edge_list = line_to_edges_fixture['stim_running']['rising_idx']
        expected_idx = []
        for stim_edge in stim_edge_list:
            this_idx = np.where(frame_edges >= stim_edge)[0].min()
            expected_idx.append(this_idx)
        result[edge_type] = expected_idx
    return result


def test_get_start_frames_exact(
        sync_file_fixture,
        line_to_edges_fixture,
        sync_sample_fixture,
        sync_freq_fixture):
    """
    Test case where _get_frame_offsets is expected to return
    exact matches
    """

    with SyncDataset(sync_file_fixture) as sync_data:
        start_frames = _get_start_frames(
                        data=sync_data,
                        raw_frame_times=sync_sample_fixture/sync_freq_fixture,
                        stimulus_frame_counts=[44, 55, 66, 77],
                        tolerance=0.0)
    np.testing.assert_array_equal(
            start_frames,
            line_to_edges_fixture['stim_running']['rising_idx'])


@pytest.mark.parametrize(
        "stimulus_frame_counts, expected",
        [([33, 66, 132], [12, 100, 500]),
         ([66, 132], [100, 500])])
def test_get_start_frames_skip_one(
        sync_file_fixture,
        sync_sample_fixture,
        sync_freq_fixture,
        stimulus_frame_counts,
        expected):
    """
    Test the case where one of the blocks in stim_running is erroneous
    """
    with SyncDataset(sync_file_fixture) as sync_data:
        start_frames = _get_start_frames(
                        data=sync_data,
                        raw_frame_times=sync_sample_fixture/sync_freq_fixture,
                        stimulus_frame_counts=stimulus_frame_counts,
                        tolerance=0.0)
    np.testing.assert_array_equal(
            start_frames,
            expected)


@pytest.mark.parametrize(
        "tolerance, stimulus_frame_counts, expected",
        [(0.1, [35, 61, 127], [12, 100, 500]),
         (0.05, [32, 68, 136], [12, 100, 500]),
         (0.05, [67, 96], [100, 204])])
def test_get_start_frames_tolerance(
        sync_file_fixture,
        sync_sample_fixture,
        sync_freq_fixture,
        tolerance,
        stimulus_frame_counts,
        expected):
    """
    Test that _get_start_frames correctly infers starting frames
    within tolerance
    """
    with SyncDataset(sync_file_fixture) as sync_data:
        start_frames = _get_start_frames(
                        data=sync_data,
                        raw_frame_times=sync_sample_fixture/sync_freq_fixture,
                        stimulus_frame_counts=stimulus_frame_counts,
                        tolerance=tolerance)
    np.testing.assert_array_equal(
            start_frames,
            expected)


@pytest.mark.parametrize(
        "tolerance, stimulus_frame_counts",
        [(0.1, [32, 40, 99]),
         (0.05, [33, 66, 300])])
def test_get_start_frames_tolerance_failures(
        sync_file_fixture,
        line_to_edges_fixture,
        sync_sample_fixture,
        sync_freq_fixture,
        tolerance,
        stimulus_frame_counts):
    """
    Test that _get_start_frames correctly fails when the best guess
    is outside of the specified tolerance
    """
    with SyncDataset(sync_file_fixture) as sync_data:
        with pytest.raises(RuntimeError, match="Could not find matching sync"):
            _get_start_frames(
                        data=sync_data,
                        raw_frame_times=sync_sample_fixture/sync_freq_fixture,
                        stimulus_frame_counts=stimulus_frame_counts,
                        tolerance=tolerance)


def test_get_start_frames_too_many_pkl(
        sync_file_fixture,
        sync_sample_fixture,
        sync_freq_fixture):
    """
    Test the case where you specify too many stimulus_frame_counts
    """
    with SyncDataset(sync_file_fixture) as sync_data:
        with pytest.raises(RuntimeError, match="more pkl frame count entries"):
            _get_start_frames(
                        data=sync_data,
                        raw_frame_times=sync_sample_fixture/sync_freq_fixture,
                        stimulus_frame_counts=[44, 55, 77, 100, 300, 55],
                        tolerance=0.0)


def test_user_facing_get_stims_error(
       sync_file_fixture):
    """
    Make sure that get_stim_timestamps_from_stimulus_blocks raises
    the expected error if you give it a bad raw_frame_time_direction
    """
    with pytest.raises(ValueError,
                       match="Cannot parse raw_frame_time_direction"):
        get_stim_timestamps_from_stimulus_blocks(
            stimulus_files=[DummyStim(n_frames=2),
                            DummyStim(n_frames=4)],
            sync_file=sync_file_fixture,
            raw_frame_time_lines='vsync_stim',
            raw_frame_time_direction='nonsense',
            frame_count_tolerance=0.0)


@pytest.mark.parametrize(
        "edge_type", ["rising", "falling"])
def test_user_facing_get_stim_timestamps_smoke(
        sync_file_fixture,
        edge_type,
        expected_start_frames_fixture,
        line_to_edges_fixture,
        sync_sample_fixture,
        sync_freq_fixture):
    """
    Test user-facing get_stim_timestaps_from_stimulus_blocks
    in case where the number of blocks in stim_running matches
    the expected number of stimulus blocks exactly
    """

    # note that the number of frames specified here
    # actually doesn't matter; because the number of blocks
    # found by analyzing stim_running matches the expected
    # number of blocks exactly, the start_frames of those
    # blocks will be returned
    stim_list = [DummyStim(n_frames=33),
                 DummyStim(n_frames=66),
                 DummyStim(n_frames=99),
                 DummyStim(n_frames=132)]

    result = get_stim_timestamps_from_stimulus_blocks(
               stimulus_files=stim_list,
               sync_file=sync_file_fixture,
               raw_frame_time_lines='vsync_stim',
               raw_frame_time_direction=edge_type,
               frame_count_tolerance=0.0)

    assert len(result["timestamps"]) == 4
    for ii, (this_array,
             this_start_frame,
             this_stim) in enumerate(zip(result["timestamps"],
                                         result["start_frames"],
                                         stim_list)):
        raw_idx = line_to_edges_fixture['vsync_stim'][f'{edge_type}_idx']
        raw_times = sync_sample_fixture[raw_idx]/sync_freq_fixture
        idx0 = expected_start_frames_fixture[edge_type][ii]
        expected = raw_times[idx0: idx0+this_stim.num_frames]
        np.testing.assert_array_equal(this_array, expected)
        assert this_start_frame == expected_start_frames_fixture[edge_type][ii]


@pytest.mark.parametrize(
        "edge_type, stim_frame_inputs, tolerance, expected_idx",
        [('rising', (11, 33, 44), 0.0, [0, 2, 3]),
         ('falling', (11, 33, 44), 0.0, [0, 2, 3]),
         ('rising', (21, 34, 45), 0.05, [1, 2, 3]),
         ('falling', (31, 47), 0.1, [2, 3])])
def test_user_facing_get_stim_timestamps(
        sync_file_fixture,
        edge_type,
        stim_frame_inputs,
        tolerance,
        expected_idx,
        expected_start_frames_fixture,
        line_to_edges_fixture,
        sync_sample_fixture,
        sync_freq_fixture):
    """
    Test the user-facing get_start_frames_from_stimulus_blocks
    in cases of differing stimulus specifications and tolerances
    """

    stim_list = [DummyStim(n_frames=n) for n in stim_frame_inputs]
    expected_start = [expected_start_frames_fixture[edge_type][idx]
                      for idx in expected_idx]

    result = get_stim_timestamps_from_stimulus_blocks(
                stimulus_files=stim_list,
                sync_file=sync_file_fixture,
                raw_frame_time_lines='vsync_stim',
                raw_frame_time_direction=edge_type,
                frame_count_tolerance=tolerance)

    assert len(result["timestamps"]) == len(expected_idx)
    for ii, (this_array,
             this_start_frame,
             this_stim) in enumerate(zip(result["timestamps"],
                                         result["start_frames"],
                                         stim_list)):
        raw_idx = line_to_edges_fixture['vsync_stim'][f'{edge_type}_idx']
        raw_times = sync_sample_fixture[raw_idx]/sync_freq_fixture
        idx0 = expected_start[ii]
        expected = raw_times[idx0: idx0+this_stim.num_frames]
        np.testing.assert_array_equal(this_array, expected)
        assert this_start_frame == expected_start[ii]
