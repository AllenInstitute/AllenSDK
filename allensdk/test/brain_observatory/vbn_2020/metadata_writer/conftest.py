import pytest
import numpy as np
import pandas as pd
import datetime
import pathlib
import tempfile


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
    this_date = datetime.datetime(year=2020, month=6, day=7)
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

    this_date = datetime.datetime(year=1998, month=3, day=14)
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

    yield output

    helper_functions.windows_safe_cleanup_dir(
            dir_path=pathlib.Path(tmp_dir))


@pytest.fixture(scope='session')
def some_files_fixture(
        tmp_path_factory,
        helper_functions):
    """
    Create some temporary files; return a list of paths to them
    """
    tmpdir = pathlib.Path(
                tmp_path_factory.mktemp('id_generator'))
    path_list = []
    for idx in range(4):
        this_path = tmpdir / f'silly_file_{idx}.nwb'
        with open(this_path, 'w') as out_file:
            out_file.write(f'this is file {idx}')
        path_list.append(this_path)

    yield path_list
    helper_functions.windows_safe_cleanup_dir(dir_path=pathlib.Path(tmpdir))


@pytest.fixture
def session_table_fixture(
        some_files_fixture):
    """
    A session_table suitable to be run through
    add_file_path_to_session_table with some_files_fixture
    the expected files to be added.

    There will be some rows that have non-existent files
    """
    rng = np.random.default_rng(66712)
    input_data = []
    for file_path in some_files_fixture:
        file_name = file_path.name
        file_idx = file_name.split('_')[-1].split('.')[0]
        file_idx = int(file_idx)
        element = {'file_index': file_idx,
                   'some_data': float(rng.random())}
        input_data.append(element)

    input_data.append({'file_index': 11232, 'some_data': float(rng.random())})
    input_data.append({'file_index': 4455, 'some_data': float(rng.random())})
    return pd.DataFrame(data=input_data)
