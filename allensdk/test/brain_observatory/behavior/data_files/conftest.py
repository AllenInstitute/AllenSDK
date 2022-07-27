import pytest
import pathlib
import tempfile
import pandas as pd

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile,
    MappingStimulusFile,
    ReplayStimulusFile)


@pytest.fixture
def behavior_pkl_fixture(
        tmp_path_factory,
        helper_functions):
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

    helper_functions.windows_safe_cleanup(file_path=pkl_path)


@pytest.fixture
def general_pkl_fixture(
        tmp_path_factory,
        helper_functions):
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

    helper_functions.windows_safe_cleanup(file_path=pkl_path)


@pytest.fixture
def behavior_stim_fixture(
        behavior_pkl_fixture):
    """
    A BehaviorStimulusFile
    """
    return BehaviorStimulusFile.from_json(
        dict_repr={"behavior_stimulus_file":
                   behavior_pkl_fixture["path"]})


@pytest.fixture
def replay_stim_fixture(
        general_pkl_fixture):
    """
    A ReplayStimulusFile
    """
    return ReplayStimulusFile.from_json(
        dict_repr={"replay_stimulus_file":
                   general_pkl_fixture["path"]})


@pytest.fixture
def mapping_stim_fixture(
        general_pkl_fixture):
    """
    A MappingStimulusFile
    """
    return MappingStimulusFile.from_json(
        dict_repr={"mapping_stimulus_file":
                   general_pkl_fixture["path"]})
