import pytest
from itertools import product
import copy
import numpy as np
import tempfile
import pathlib
import pandas as pd
from unittest.mock import patch

from allensdk.brain_observatory.behavior.\
    data_objects.running_speed.running_processing import (
        get_running_df)

from allensdk.brain_observatory.ecephys.\
    data_objects.running_speed.multi_stim_running_processing import (
        _extract_dx_info,
        _get_behavior_frame_count,
        _get_frame_count,
        _get_frame_counts,
        _get_frame_times,
        _get_stimulus_starts_and_ends,
        _merge_dx_data,
        multi_stim_running_df_from_raw_data)


@pytest.mark.parametrize(
        "index_bounds, use_lowpass, zscore",
        product(((30, 70), (10, 60)),
                (True, False),
                (5.0, 10.0)))
def test_extract_dx_start_end(
        basic_running_stim_file_fixture,
        index_bounds,
        use_lowpass,
        zscore,
        tmp_path_factory):
    """
    Test that _extract_dx_info behaves like get_running_df with
    start_index and end_index handled correctly
    """
    tmpdir = tmp_path_factory.mktemp('extract_dx_test')

    stim_file = basic_running_stim_file_fixture

    n_time = len(stim_file['items']['behavior']['encoders'][0]['dx'])
    time_array = np.linspace(0., 10., n_time)

    expected_stim = copy.deepcopy(stim_file)
    for key in ('dx', 'vsig', 'vin'):
        raw = expected_stim['items']['behavior']['encoders'][0].pop(key)
        expected_stim['items']['behavior']['encoders'][0][key] = raw[
                                                               index_bounds[0]:
                                                               index_bounds[1]]

    expected = get_running_df(
                  data=expected_stim,
                  time=time_array[index_bounds[0]:index_bounds[1]],
                  lowpass=use_lowpass,
                  zscore_threshold=zscore)

    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pd.to_pickle(expected_stim, pkl_path)

    actual = _extract_dx_info(
                frame_times=time_array,
                start_index=index_bounds[0],
                end_index=index_bounds[1],
                pkl_path=str(pkl_path.resolve().absolute()),
                zscore_threshold=zscore,
                use_lowpass_filter=use_lowpass)

    pd.testing.assert_frame_equal(actual, expected)

    if pkl_path.exists():
        pkl_path.unlink()


def test_extract_dx_time_mismatch(
        basic_running_stim_file_fixture,
        tmp_path_factory):
    """
    Test that an exception gets thrown if
    start_index, end_index do not map frame_times
    to the correct length
    """
    index_bounds = (10, 60)

    tmpdir = tmp_path_factory.mktemp('extract_dx_test')

    stim_file = basic_running_stim_file_fixture

    n_time = len(stim_file['items']['behavior']['encoders'][0]['dx'])
    time_array = np.linspace(0., 10., n_time)

    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pd.to_pickle(stim_file, pkl_path)

    with pytest.raises(ValueError, match="length of v_in"):
        _extract_dx_info(
                frame_times=time_array,
                start_index=index_bounds[0],
                end_index=index_bounds[1],
                pkl_path=str(pkl_path.resolve().absolute()),
                zscore_threshold=10.0,
                use_lowpass_filter=True)

    if pkl_path.exists():
        pkl_path.unlink()


@pytest.mark.parametrize('frame_count', [10, 12, 27])
def test_get_behavior_frame_count(frame_count, tmp_path_factory):
    """
    Test that _get_behavior_frame_count measure the correct value
    """
    tmpdir = tmp_path_factory.mktemp('behavior_frame_count_test')
    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    data = {'items': {'behavior': {'intervalsms': list(range(frame_count-1))}}}
    pd.to_pickle(data, pkl_path)
    actual = _get_behavior_frame_count(str(pkl_path.resolve().absolute()))
    assert actual == frame_count
    if pkl_path.exists():
        pkl_path.unlink()


@pytest.mark.parametrize('frame_count', [10, 12, 27])
def test_get_frame_count(frame_count, tmp_path_factory):
    """
    Test that _get_frame_count measure the correct value
    """
    tmpdir = tmp_path_factory.mktemp('frame_count_test')
    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    data = {'intervalsms': list(range(frame_count-1))}
    pd.to_pickle(data, pkl_path)
    actual = _get_frame_count(str(pkl_path.resolve().absolute()))
    assert actual == frame_count
    if pkl_path.exists():
        pkl_path.unlink()


def test_get_frame_counts(
        stimulus_file_frame_fixture):
    """
    Test that _get_frame_counts returns the right
    frame counts in the right order
    """
    (frame_count_lookup,
     pkl_path_lookup) = stimulus_file_frame_fixture

    actual = _get_frame_counts(
           behavior_pkl_path=pkl_path_lookup['behavior'],
           mapping_pkl_path=pkl_path_lookup['mapping'],
           replay_pkl_path=pkl_path_lookup['replay'])

    assert actual[0] == frame_count_lookup['behavior']
    assert actual[1] == frame_count_lookup['mapping']
    assert actual[2] == frame_count_lookup['replay']
    assert len(actual) == 3


def test_get_frame_times():
    """
    Test that _get_frame_times invokes sync_data.get_edges to find the
    rising edges with units='seconds'
    """

    def dummy_init(self, sync_path):
        pass

    def dummy_get_edges(self, kind, frame_keys, units=None):
        if kind != "rising":
            msg = f"asked for {kind} edges; must be rising"
            raise RuntimeError(msg)
        if units != "seconds":
            msg = f"units must be 'seconds'; gave {units}"
            raise RuntimeError(msg)
        return None

    with patch('allensdk.brain_observatory.sync_dataset.Dataset.__init__',
               new=dummy_init):
        with patch('allensdk.brain_observatory.sync_dataset.Dataset.get_edges',
                   new=dummy_get_edges):
            _get_frame_times('nonsense')


@pytest.mark.parametrize('start_frame', [1, 5, 9])
def test_get_stimulus_starts_ends(
        stimulus_file_frame_fixture,
        start_frame):
    """
    Test that _get_stimulus_starts_and_ends returns the correct frame
    indices in the correct order
    """
    (frame_ct,
     pkl_path_lookup) = stimulus_file_frame_fixture

    actual = _get_stimulus_starts_and_ends(
           behavior_pkl_path=pkl_path_lookup['behavior'],
           mapping_pkl_path=pkl_path_lookup['mapping'],
           replay_pkl_path=pkl_path_lookup['replay'],
           behavior_start_frame=start_frame)

    expected = (
        start_frame,
        frame_ct['behavior'],
        frame_ct['behavior']+frame_ct['mapping'],
        frame_ct['behavior']+frame_ct['mapping']+frame_ct['replay'])

    assert actual == expected


def test_get_stimulus_starts_ends_error(
        stimulus_file_frame_fixture):
    """
    Test that _get_stimulus_starts_and_ends raises an error
    if maping_start <= behavior_start
    """
    (frame_ct,
     pkl_path_lookup) = stimulus_file_frame_fixture

    with pytest.raises(RuntimeError, match="behavior_start_frame"):
        _get_stimulus_starts_and_ends(
              behavior_pkl_path=pkl_path_lookup['behavior'],
              mapping_pkl_path=pkl_path_lookup['mapping'],
              replay_pkl_path=pkl_path_lookup['replay'],
              behavior_start_frame=frame_ct['behavior'])

    with pytest.raises(RuntimeError, match="behavior_start_frame"):
        _get_stimulus_starts_and_ends(
              behavior_pkl_path=pkl_path_lookup['behavior'],
              mapping_pkl_path=pkl_path_lookup['mapping'],
              replay_pkl_path=pkl_path_lookup['replay'],
              behavior_start_frame=frame_ct['behavior']+6)


@pytest.mark.parametrize('start_frame', [0, 15])
def test_merge_dx_data(merge_data_fixture,
                       start_frame):
    """
    Test that _merge_dx_data correctly merges dataframes
    """

    frame_times = np.linspace(0.,
                              10.,
                              merge_data_fixture['n_timesteps']+start_frame)

    (velocity_df,
     raw_df) = _merge_dx_data(
        mapping_velocities=merge_data_fixture['mapping']['dataframe'],
        behavior_velocities=merge_data_fixture['behavior']['dataframe'],
        replay_velocities=merge_data_fixture['replay']['dataframe'],
        frame_times=frame_times,
        behavior_start_frame=start_frame)

    assert set(velocity_df.columns) == set(['velocity',
                                            'net_rotation',
                                            'frame_indexes',
                                            'frame_time'])

    assert set(raw_df.columns) == set(['vsig', 'vin', 'frame_time', 'dx'])

    # make sure that velocity_df skipped some timesteps
    assert len(velocity_df.velocity.values) < len(raw_df.dx.values)

    # check that velocity_df has the expected values, having
    # skipped the right timesteps
    for in_key, out_key in zip(('dx', 'speed'),
                               ('net_rotation', 'velocity')):
        expected = []
        for pkl_key in ('behavior', 'mapping', 'replay'):
            kept = merge_data_fixture[pkl_key]['kept_mask']
            expected.append(merge_data_fixture[pkl_key][in_key][kept])
        expected = np.concatenate(expected)
        np.testing.assert_array_equal(velocity_df[out_key].values, expected)

    # check contents of frame_time and frame_indexes
    global_kept = np.concatenate([merge_data_fixture['behavior']['kept_mask'],
                                  merge_data_fixture['mapping']['kept_mask'],
                                  merge_data_fixture['replay']['kept_mask']])
    np.testing.assert_array_equal(
            frame_times[start_frame:][global_kept],
            velocity_df.frame_time.values)

    np.testing.assert_array_equal(
            np.arange(start_frame,
                      merge_data_fixture['n_timesteps']+start_frame,
                      dtype=int)[global_kept],
            velocity_df.frame_indexes.values)

    # check contents of raw_df
    for in_key, out_key in zip(('dx', 'v_in', 'v_sig'),
                               ('dx', 'vin', 'vsig')):
        expected = []
        for pkl_key in ('behavior', 'mapping', 'replay'):
            expected.append(merge_data_fixture[pkl_key][in_key])
        expected = np.concatenate(expected)
        np.testing.assert_array_equal(expected, raw_df[out_key].values)

    np.testing.assert_array_equal(raw_df.frame_time.values,
                                  frame_times[start_frame:])


@pytest.mark.parametrize("start_frame", [0, ])
def test_multi_stim_running_df_from_raw_data(
        start_frame,
        behavior_pkl_fixture,
        replay_pkl_fixture,
        mapping_pkl_fixture):
    """
    test that multi_stim_running_df_from_raw_data
    can properly process stimulus pickle files
    """

    use_lowpass = True
    zscore = 10.0

    def dummy_get_frame_times(sync_path=None):
        nt = behavior_pkl_fixture['n_frames']
        nt += replay_pkl_fixture['n_frames']
        nt += mapping_pkl_fixture['n_frames']
        nt += start_frame
        return np.linspace(0., 10., nt)

    to_replace = ('allensdk.brain_observatory.ecephys.data_objects.'
                  'running_speed.multi_stim_running_processing.'
                  '_get_frame_times')
    with patch(to_replace, new=dummy_get_frame_times):
        multi_stim_running_df_from_raw_data(
            sync_path='garbage',
            behavior_pkl_path=behavior_pkl_fixture['path_to_pkl'],
            mapping_pkl_path=mapping_pkl_fixture['path_to_pkl'],
            replay_pkl_path=replay_pkl_fixture['path_to_pkl'],
            use_lowpass_filter=use_lowpass,
            zscore_threshold=zscore,
            behavior_start_frame=start_frame)
