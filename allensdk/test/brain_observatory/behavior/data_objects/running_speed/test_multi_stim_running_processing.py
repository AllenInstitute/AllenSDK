import pytest
from itertools import product
import copy
import numpy as np
import tempfile
import pathlib
import pandas as pd

from allensdk.brain_observatory.behavior.\
    data_objects.running_speed.running_processing import (
        get_running_df)

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile,
    ReplayStimulusFile,
    MappingStimulusFile)

from allensdk.brain_observatory.behavior.\
    data_objects.running_speed.multi_stim_running_processing import (
        _extract_dx_info,
        _merge_dx_data,
        multi_stim_running_df_from_raw_data)


@pytest.mark.parametrize(
        "use_lowpass, zscore",
        product((True, False),
                (5.0, 10.0)))
def test_extract_dx_basic(
        basic_running_stim_file_fixture,
        use_lowpass,
        zscore,
        tmp_path_factory,
        helper_functions):
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
        expected_stim['items']['behavior']['encoders'][0][key] = raw

    expected = get_running_df(
                  data=expected_stim,
                  time=time_array,
                  lowpass=use_lowpass,
                  zscore_threshold=zscore)

    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pd.to_pickle(expected_stim, pkl_path)

    class DummyStimFile(object):
        def __init__(self, pkl_path):
            self.data = pd.read_pickle(pkl_path)

    actual = _extract_dx_info(
                frame_times=time_array,
                stimulus_file=DummyStimFile(pkl_path.resolve().absolute()),
                zscore_threshold=zscore,
                use_lowpass_filter=use_lowpass)

    pd.testing.assert_frame_equal(actual, expected)

    helper_functions.windows_safe_cleanup(file_path=pkl_path)


def test_extract_dx_time_mismatch(
        basic_running_stim_file_fixture,
        tmp_path_factory,
        helper_functions):
    """
    Test that an exception gets thrown if frame_times is not of the
    correct length
    """

    tmpdir = tmp_path_factory.mktemp('extract_dx_test')

    stim_file = basic_running_stim_file_fixture

    n_time = len(stim_file['items']['behavior']['encoders'][0]['dx'])
    time_array = np.linspace(0., 10., n_time-5)

    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pd.to_pickle(stim_file, pkl_path)

    class DummyStimFile(object):
        def __init__(self, pkl_path):
            self.data = pd.read_pickle(pkl_path)

    with pytest.raises(ValueError, match="length of v_in"):
        _extract_dx_info(
                frame_times=time_array,
                stimulus_file=DummyStimFile(pkl_path.resolve().absolute()),
                zscore_threshold=10.0,
                use_lowpass_filter=True)

    helper_functions.windows_safe_cleanup(file_path=pkl_path)


@pytest.mark.parametrize('start_frame', [0, 15])
def test_merge_dx_data(merge_data_fixture,
                       start_frame):
    """
    Test that _merge_dx_data correctly merges dataframes
    """

    frame_times = np.linspace(0.,
                              10.,
                              merge_data_fixture['n_timesteps'])

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
            frame_times[global_kept],
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
                                  frame_times)


@pytest.mark.parametrize("start_frame", [0, ])
def test_multi_stim_running_df_from_raw_data(
        start_frame,
        behavior_pkl_fixture,
        replay_pkl_fixture,
        mapping_pkl_fixture,
        sync_path_fixture):
    """
    test that multi_stim_running_df_from_raw_data
    can properly process stimulus pickle files
    """

    use_lowpass = True
    zscore = 10.0

    b_stim = BehaviorStimulusFile.from_json(
                dict_repr={'behavior_stimulus_file':
                           behavior_pkl_fixture['path_to_pkl']})

    r_stim = ReplayStimulusFile.from_json(
                dict_repr={'replay_stimulus_file':
                           replay_pkl_fixture['path_to_pkl']})

    m_stim = MappingStimulusFile.from_json(
                dict_repr={'mapping_stimulus_file':
                           mapping_pkl_fixture['path_to_pkl']})

    (velocities_df,
     raw_df) = multi_stim_running_df_from_raw_data(
                sync_path=sync_path_fixture,
                behavior_stimulus_file=b_stim,
                mapping_stimulus_file=m_stim,
                replay_stimulus_file=r_stim,
                use_lowpass_filter=use_lowpass,
                zscore_threshold=zscore,
                behavior_start_frame=start_frame)
