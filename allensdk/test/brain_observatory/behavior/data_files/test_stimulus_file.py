from typing import Tuple
from pathlib import Path
import tempfile
import pickle
import datetime
from unittest.mock import create_autospec

import pytest

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import (
    BehaviorStimulusFile,
    ReplayStimulusFile,
    MappingStimulusFile)

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    StimulusFileLookup,
    stimulus_lookup_from_json)

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BEHAVIOR_STIMULUS_FILE_QUERY_TEMPLATE
)


@pytest.fixture
def stimulus_file_fixture(request, tmp_path) -> Tuple[Path, dict]:
    default_stim_pkl_data = {"a": 1, "b": 2, "c": 3}
    stim_pkl_data = request.param.get("pkl_data", default_stim_pkl_data)
    stim_pkl_filename = request.param.get("filename", "test_stimulus_file.pkl")

    stim_pkl_path = tmp_path / stim_pkl_filename
    with stim_pkl_path.open('wb') as f:
        pickle.dump(stim_pkl_data, f)

    return (stim_pkl_path, stim_pkl_data)


@pytest.mark.parametrize("stimulus_file_fixture", [
    ({"pkl_data": {"a": 42, "b": 7}}),
    ({"pkl_data": {"slightly_more_complex": [1, 2, 3, 4]}})
], indirect=["stimulus_file_fixture"])
def test_stimulus_file_from_json(stimulus_file_fixture):
    stim_pkl_path, stim_pkl_data = stimulus_file_fixture

    # Basic test case
    input_json_dict = {"behavior_stimulus_file": str(stim_pkl_path)}
    stimulus_file = BehaviorStimulusFile.from_json(input_json_dict)
    assert stimulus_file.data == stim_pkl_data

    # Now test caching by deleting the stimulus_file
    stim_pkl_path.unlink()
    stimulus_file_cached = BehaviorStimulusFile.from_json(input_json_dict)
    assert stimulus_file_cached.data == stim_pkl_data


@pytest.mark.parametrize("stimulus_file_fixture, behavior_session_id", [
    ({"pkl_data": {"a": 42, "b": 7}}, 12),
    ({"pkl_data": {"slightly_more_complex": [1, 2, 3, 4]}}, 8)
], indirect=["stimulus_file_fixture"])
def test_stimulus_file_from_lims(stimulus_file_fixture, behavior_session_id):
    stim_pkl_path, stim_pkl_data = stimulus_file_fixture

    mock_db_conn = create_autospec(PostgresQueryMixin, instance=True)

    # Basic test case
    mock_db_conn.fetchone.return_value = str(stim_pkl_path)
    stimulus_file = BehaviorStimulusFile.from_lims(
                           mock_db_conn,
                           behavior_session_id)
    assert stimulus_file.data == stim_pkl_data

    # Now test caching by deleting stimulus_file and also asserting db
    # `fetchone` called only once
    stim_pkl_path.unlink()
    stimfile_cached = BehaviorStimulusFile.from_lims(
                            mock_db_conn,
                            behavior_session_id)
    assert stimfile_cached.data == stim_pkl_data

    query = BEHAVIOR_STIMULUS_FILE_QUERY_TEMPLATE.format(
        behavior_session_id=behavior_session_id
    )

    mock_db_conn.fetchone.assert_called_once_with(query, strict=True)


def test_behavior_num_frames(
        behavior_pkl_fixture):
    """
    Test that BehaviorStimulusFile.num_frames returns the
    expected result
    """
    str_path = str(behavior_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"behavior_stimulus_file": str_path}
    beh_stim = BehaviorStimulusFile.from_json(dict_repr=dict_repr)
    assert beh_stim.num_frames == behavior_pkl_fixture['expected_frames']


def test_replay_num_frames(
        general_pkl_fixture):
    """
    Test that ReplayStimulusFile.num_frames returns the
    expected result
    """
    str_path = str(general_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"replay_stimulus_file": str_path}
    rep_stim = ReplayStimulusFile.from_json(dict_repr=dict_repr)
    assert rep_stim.num_frames == general_pkl_fixture['expected_frames']


def test_mapping_num_frames(
        general_pkl_fixture):
    """
    Test that MappingStimulusFile.num_frames returns the
    expected result
    """
    str_path = str(general_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"mapping_stimulus_file": str_path}
    map_stim = MappingStimulusFile.from_json(dict_repr=dict_repr)
    assert map_stim.num_frames == general_pkl_fixture['expected_frames']


def test_malformed_behavior_pkl(
        general_pkl_fixture):
    """
    Test that the correct error is raised when a behavior pickle file
    is mal-formed
    """
    str_path = str(general_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"behavior_stimulus_file": str_path}
    stim = BehaviorStimulusFile.from_json(dict_repr=dict_repr)
    with pytest.raises(RuntimeError,
                       match="When getting num_frames from"):
        _ = stim.num_frames


def test_stimulus_file_lookup(
        behavior_stim_fixture,
        mapping_stim_fixture,
        replay_stim_fixture):
    """
    Smoke test of StimulusFileLookup
    """
    lookup = StimulusFileLookup()

    with pytest.raises(ValueError, match="has no BehaviorStimulusFile"):
        lookup.behavior_stimulus_file

    with pytest.raises(ValueError, match="has no ReplayStimulusFile"):
        lookup.replay_stimulus_file

    with pytest.raises(ValueError, match="has no MappingStimulusFile"):
        lookup.mapping_stimulus_file

    lookup.behavior_stimulus_file = behavior_stim_fixture
    lookup.mapping_stimulus_file = mapping_stim_fixture
    lookup.replay_stimulus_file = replay_stim_fixture

    assert isinstance(lookup.behavior_stimulus_file, BehaviorStimulusFile)
    assert lookup.behavior_stimulus_file.data == behavior_stim_fixture.data

    assert isinstance(lookup.replay_stimulus_file, ReplayStimulusFile)
    assert lookup.replay_stimulus_file.data == replay_stim_fixture.data

    assert isinstance(lookup.mapping_stimulus_file, MappingStimulusFile)
    assert lookup.mapping_stimulus_file.data == mapping_stim_fixture.data


@pytest.mark.parametrize(
    "to_set, set_from",
    [('behavior', 'replay'),
     ('behavior', 'mapping'),
     ('replay', 'behavior'),
     ('replay', 'mapping'),
     ('mapping', 'behavior'),
     ('mapping', 'replay')])
def test_stimulus_file_lookup_errors(
        behavior_stim_fixture,
        mapping_stim_fixture,
        replay_stim_fixture,
        to_set,
        set_from):
    """
    Test that errors get raised when the wrong StimulusFile
    is passed into lookup
    """
    if set_from == 'behavior':
        src = behavior_stim_fixture
    elif set_from == 'mapping':
        src = mapping_stim_fixture
    elif set_from == 'replay':
        src = replay_stim_fixture

    lookup = StimulusFileLookup()
    if to_set == 'behavior':
        with pytest.raises(ValueError, match="should be BehaviorStimulusFile"):
            lookup.behavior_stimulus_file = src
    elif to_set == 'replay':
        with pytest.raises(ValueError, match="should be ReplayStimulusFile"):
            lookup.replay_stimulus_file = src
    else:
        with pytest.raises(ValueError, match="should be MappingStimulusFile"):
            lookup.mapping_stimulus_file = src


@pytest.mark.parametrize(
    "to_use",
    [('behavior',),
     ('mapping',),
     ('replay',),
     ('behavior', 'replay'),
     ('behavior', 'mapping'),
     ('mapping', 'replay'),
     ('behavior', 'mapping', 'replay')])
def test_stimulus_lookup_from_json(
        behavior_stim_fixture,
        mapping_stim_fixture,
        replay_stim_fixture,
        general_pkl_fixture,
        behavior_pkl_fixture,
        to_use):
    """
    Smoke test for stimulus_lookup_from_json
    """

    dict_repr = dict()
    if 'behavior' in to_use:
        dict_repr['behavior_stimulus_file'] = behavior_pkl_fixture['path']
    if 'mapping' in to_use:
        dict_repr['mapping_stimulus_file'] = general_pkl_fixture['path']
    if 'replay' in to_use:
        dict_repr['replay_stimulus_file'] = general_pkl_fixture['path']

    lookup = stimulus_lookup_from_json(dict_repr=dict_repr)
    assert isinstance(lookup, StimulusFileLookup)

    if 'behavior' in to_use:
        assert isinstance(lookup.behavior_stimulus_file, BehaviorStimulusFile)
        assert lookup.behavior_stimulus_file.data == behavior_stim_fixture.data
    else:
        with pytest.raises(ValueError, match="has no BehaviorStimulusFile"):
            lookup.behavior_stimulus_file

    if 'mapping' in to_use:
        assert isinstance(lookup.mapping_stimulus_file, MappingStimulusFile)
        assert lookup.mapping_stimulus_file.data == mapping_stim_fixture.data
    else:
        with pytest.raises(ValueError, match="has no MappingStimulusFile"):
            lookup.mapping_stimulus_file

    if 'replay' in to_use:
        assert isinstance(lookup.replay_stimulus_file, ReplayStimulusFile)
        assert lookup.replay_stimulus_file.data == replay_stim_fixture.data
    else:
        with pytest.raises(ValueError, match="has no ReplayStimulusFile"):
            lookup.replay_stimulus_file


@pytest.mark.parametrize(
    "cl_value_in, param_value_in, expected_value, expected_error_msg",
    [(None, 'junk', 'junk', None),
     ('silly', None, 'silly', None),
     ('fun', 'fun', 'fun', None),
     (None, None, None, 'Could not find stage in pickle file'),
     ('something', 'else', None, 'Conflicting session_types')])
def test_behavior_session_type(
        cl_value_in,
        param_value_in,
        expected_value,
        expected_error_msg,
        tmp_path_factory,
        helper_functions):

    this_tmp_dir = tmp_path_factory.mktemp('beh_session_type')
    this_tmp_path = Path(
                        tempfile.mkstemp(
                            dir=this_tmp_dir,
                            suffix='.pkl')[1])

    stim_dict = {'items':
                 {'behavior': dict()}}
    if cl_value_in is not None:
        stim_dict['items']['behavior']['cl_params'] = {'stage': cl_value_in}
    if param_value_in is not None:
        stim_dict['items']['behavior']['params'] = {'stage': param_value_in}

    with open(this_tmp_path, 'wb') as out_file:
        pickle.dump(stim_dict, out_file)

    stim_file = BehaviorStimulusFile(this_tmp_path)

    if expected_error_msg is None:
        actual = stim_file.session_type
        assert actual == expected_value
    else:
        with pytest.raises(RuntimeError, match=expected_error_msg):
            stim_file.session_type

    helper_functions.windows_safe_cleanup_dir(
                dir_path=Path(this_tmp_dir))


def test_behavior_start_time(
        tmp_path_factory,
        helper_functions):
    """
    Test BehaviorStimulusFile.date_of_acquisition
    """
    this_tmp_dir = tmp_path_factory.mktemp('date_of_acq')

    expected = datetime.datetime(1972, 3, 14, 23, 30, 41)

    good_data = {'start_time': expected}
    this_tmp_path = Path(
                        tempfile.mkstemp(
                            dir=this_tmp_dir,
                            suffix='.pkl')[1])

    with open(this_tmp_path, 'wb') as out_file:
        pickle.dump(good_data, out_file)

    stim_file = BehaviorStimulusFile(this_tmp_path)
    assert stim_file.date_of_acquisition == expected

    bad_data = {'nothing': 'at all'}
    other_tmp_path = Path(
                        tempfile.mkstemp(
                            dir=this_tmp_dir,
                            suffix='.pkl')[1])
    with open(other_tmp_path, 'wb') as out_file:
        pickle.dump(bad_data, out_file)

    stim_file = BehaviorStimulusFile(other_tmp_path)
    with pytest.raises(KeyError, match="No \'start_time\' listed in pickle"):
        stim_file.date_of_acquisition

    helper_functions.windows_safe_cleanup_dir(
                dir_path=Path(this_tmp_dir))
