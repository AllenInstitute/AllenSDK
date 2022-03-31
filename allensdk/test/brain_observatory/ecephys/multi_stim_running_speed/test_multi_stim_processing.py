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

from allensdk.brain_observatory.ecephys.\
    data_objects.running_speed.multi_stim_running_processing import (
        _extract_dx_info,
        _get_behavior_frame_count,
        _get_frame_count,
        _get_frame_counts)


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


def test_get_frame_counts(tmp_path_factory):
    """
    Test that _get_frame_counts returns the right
    frame counts in the right order
    """
    tmpdir = tmp_path_factory.mktemp('all_frame_count_test')
    pkl_path_lookup = dict()
    pkl_path_lookup['behavior'] = pathlib.Path(
                          tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pkl_path_lookup['mapping'] = pathlib.Path(
                          tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    pkl_path_lookup['replay'] = pathlib.Path(
                          tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    frame_count_lookup = {'behavior': 13, 'mapping': 44, 'replay': 76}

    data = {'items':
            {'behavior':
             {'intervalsms':
              list(range(frame_count_lookup['behavior']-1))}}}
    pd.to_pickle(data, pkl_path_lookup['behavior'])

    for key in ('mapping', 'replay'):
        data = {'intervalsms': list(range(frame_count_lookup[key]-1))}
        pd.to_pickle(data, pkl_path_lookup[key])

    actual = _get_frame_counts(
       behavior_pkl_path=str(pkl_path_lookup['behavior'].resolve().absolute()),
       mapping_pkl_path=str(pkl_path_lookup['mapping'].resolve().absolute()),
       replay_pkl_path=str(pkl_path_lookup['replay'].resolve().absolute()))

    assert actual[0] == frame_count_lookup['behavior']
    assert actual[1] == frame_count_lookup['mapping']
    assert actual[2] == frame_count_lookup['replay']
    assert len(actual) == 3

    for key in pkl_path_lookup:
        pth = pkl_path_lookup[key]
        if pth.exists():
            pth.unlink()
