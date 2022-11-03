import pathlib

import numpy as np
import pandas as pd
import pytest


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
def metadata_table_fixture(
        some_files_fixture):
    """
    A metadata_table suitable to be run through
    add_file_path_to_metadata_table with some_files_fixture
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
