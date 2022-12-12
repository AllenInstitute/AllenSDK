import pytest
import pandas as pd
import datetime
import pathlib
import tempfile

from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisition


@pytest.fixture
def smoketest_config_fixture():
    """
    config parameters for on-prem metadata writer smoketest
    """
    config = {
      "ecephys_session_id_list": [1115077618, 1081429294],
      "probes_to_skip": [{"session": 1115077618, "probe": "probeC"}]
    }
    return config


@pytest.fixture
def smoketest_with_failed_sessions_config_fixture():
    """
    config parameters for on-prem metadata writer smoketest
    """
    config = {
      "ecephys_session_id_list": [1051155866],
      "failed_ecephys_session_id_list": [1050962145],
      "probes_to_skip": [{"session": 1115077618, "probe": "probeC"}]
    }
    return config


@pytest.fixture
def patching_pickle_file_fixture(
        helper_functions,
        tmp_path_factory):
    """
    Write mock data to some stimulus pickles.
    Return a dict mapping behavior_session_id to
        The path to the pickle
        The session_type stored in the pickle
        The date_of_acquisition stored in the pickle
    """
    tmp_dir = tmp_path_factory.mktemp('patching_pickles')

    output = dict()
    this_date = DateOfAcquisition(
        datetime.datetime(year=2020, month=6, day=7)).value
    this_stage = 'first_stage'
    pkl_data = {'start_time': this_date,
                'items':
                {'behavior':
                 {'params': {'stage': this_stage}}}}
    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmp_dir, suffix='.pkl')[1])
    pd.to_pickle(pkl_data, pkl_path)
    output[1123] = {'pkl_path': pkl_path,
                    'date_of_acquisition': this_date,
                    'session_type': this_stage}

    this_date = DateOfAcquisition(
        datetime.datetime(year=1998, month=3, day=14)).value
    this_stage = 'second_stage'
    pkl_data = {'start_time': this_date,
                'items':
                {'behavior':
                 {'params': {'stage': this_stage}}}}
    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmp_dir, suffix='.pkl')[1])
    pd.to_pickle(pkl_data, pkl_path)
    output[5813] = {'pkl_path': pkl_path,
                    'date_of_acquisition': this_date,
                    'session_type': this_stage}

    this_date = DateOfAcquisition(
        datetime.datetime(year=1974, month=7, day=22)).value
    this_stage = 'third_stage'
    pkl_data = {'start_time': this_date,
                'items':
                {'behavior':
                 {'params': {'stage': this_stage}}}}
    pkl_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmp_dir, suffix='.pkl')[1])
    pd.to_pickle(pkl_data, pkl_path)
    output[2134] = {'pkl_path': pkl_path,
                    'date_of_acquisition': this_date,
                    'session_type': this_stage}

    yield output

    helper_functions.windows_safe_cleanup_dir(
            dir_path=pathlib.Path(tmp_dir))
