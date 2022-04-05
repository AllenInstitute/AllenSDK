import pytest
import numpy as np
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset
from allensdk.brain_observatory.sync_stim_aligner import (
    _get_start_frames)


@pytest.fixture(scope='module')
def line_name_fixture():
    return ['lineA', 'stim_running', 'lineB']


@pytest.fixture(scope='module')
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
    # 44, 55, 66, 77 in the stim_running line
    this_rising = [12, 100, 204, 500]
    this_falling = [56, 155, 270, 577]
    result['stim_running'] = {'rising_idx': np.array(this_rising),
                              'falling_idx': np.array(this_falling)}
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
        [([44, 55, 77], [12, 100, 500]),
         ([55, 77], [100, 500])])
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
        [(0.1, [43, 59, 84], [12, 100, 500]),
         (0.05, [45, 53, 78], [12, 100, 500]),
         (0.05, [53, 69], [100, 204])])
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
        [(0.1, [43, 40, 84]),
         (0.05, [45, 53, 90])])
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
