import pytest
import pathlib
import tempfile
import pandas as pd


@pytest.fixture
def behavior_pkl_fixture(
        tmp_path_factory):
    """
    Write a behavior pkl file to disk.
    Return a dict containing the path to that file, as well as the
    expected number of frames associated with the pickle file.
    """
    tmpdir = tmp_path_factory.mktemp('behavior_pkl')
    pkl_path = pathlib.Path(
                   tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    nframes = 17
    result = {'items':
              {'behavior':
               {'intervalsms': list(range(nframes-1))}}}

    pd.to_pickle(result, pkl_path)
    yield {'path': pkl_path, 'expected_frames': nframes}

    try:
        if pkl_path.exists():
            pkl_path.unlink()
    except PermissionError:
        pass


@pytest.fixture
def general_pkl_fixture(
        tmp_path_factory):
    """
    Write a non-behavior stimulus pkl file to disk.
    Return a dict containing the path to that file, as well as the
    expected number of frames associated with the pickle file.
    """
    tmpdir = tmp_path_factory.mktemp('general_pkl')
    pkl_path = pathlib.Path(
                   tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1])

    nframes = 19
    result = {'intervalsms': list(range(nframes-1))}

    pd.to_pickle(result, pkl_path)
    yield {'path': pkl_path, 'expected_frames': nframes}

    try:
        if pkl_path.exists():
            pkl_path.unlink()
    except PermissionError:
        pass
