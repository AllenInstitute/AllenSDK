import pytest
import numpy as np
import pathlib
import tempfile
import pandas as pd


@pytest.fixture
def basic_running_stim_file_fixture():
    rng = np.random.default_rng()
    return {
        "items": {
            "behavior": {
                "encoders": [
                    {
                        "dx": rng.random((100,)),
                        "vsig": rng.uniform(low=0.0, high=5.1, size=(100,)),
                        "vin": rng.uniform(low=4.9, high=5.0, size=(100,)),
                    }]}}}


@pytest.fixture
def stimulus_file_frame_fixture(
        tmp_path_factory):
    """
    Writes some skeletal stimulus files (really only good for getting
    frame counts) to disk. Yields a tuple of dicts

    frame_count_lookup maps the type of file to the number
    of expected frames (type of file is 'behavior', 'mapping',
    or 'replay')

    pkl_path_lookup maps the type of file to the path to the
    temporary pickle file
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

    yield (frame_count_lookup, pkl_path_lookup)

    for key in pkl_path_lookup:
        pth = pkl_path_lookup[key]
        if pth.exists():
            pth.unlink()
