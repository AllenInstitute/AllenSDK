from typing import Tuple
from pathlib import Path
import pickle
from unittest.mock import create_autospec

import pytest

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import (
    BehaviorStimulusFile,
    ReplayStimulusFile,
    MappingStimulusFile)

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


@pytest.mark.parametrize("stimulus_file_fixture", [
    ({"filename": "test_stim_file_1.pkl"}),
    ({"filename": "mock_stim_pkl_2.pkl"})
], indirect=["stimulus_file_fixture"])
def test_stimulus_file_to_json(stimulus_file_fixture):
    stim_pkl_path, stim_pkl_data = stimulus_file_fixture

    stimulus_file = BehaviorStimulusFile(filepath=stim_pkl_path)
    obt_json = stimulus_file.to_json()
    assert obt_json == {"behavior_stimulus_file": str(stim_pkl_path)}


@pytest.mark.parametrize(
        "stimulus_class_name", ["replay", "mapping"]
)
def test_replay_mapping_round_trip(
        general_pkl_fixture,
        stimulus_class_name):
    """
    Test the round tripping of ReplayStimulusFile and MappingStimulusFile
    """
    if stimulus_class_name == "replay":
        json_key = "replay_stimulus_file"
        stimulus_class = ReplayStimulusFile
    else:
        json_key = "mapping_stimulus_file"
        stimulus_class = MappingStimulusFile

    str_path = str(general_pkl_fixture['path'].resolve().absolute())
    dict_repr = {json_key: str_path}
    stim = stimulus_class.from_json(dict_repr=dict_repr)

    new_dict_repr = stim.to_json()
    new_stim = stimulus_class.from_json(dict_repr=new_dict_repr)
    assert new_stim.data == stim.data


def test_behavior_num_frames(
        behavior_pkl_fixture):
    """
    Test that BehaviorStimulusFile.num_frames returns the
    expected result
    """
    str_path = str(behavior_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"behavior_stimulus_file": str_path}
    beh_stim = BehaviorStimulusFile.from_json(dict_repr=dict_repr)
    assert beh_stim.num_frames() == behavior_pkl_fixture['expected_frames']


def test_replay_num_frames(
        general_pkl_fixture):
    """
    Test that ReplayStimulusFile.num_frames returns the
    expected result
    """
    str_path = str(general_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"replay_stimulus_file": str_path}
    rep_stim = ReplayStimulusFile.from_json(dict_repr=dict_repr)
    assert rep_stim.num_frames() == general_pkl_fixture['expected_frames']


def test_mapping_num_frames(
        general_pkl_fixture):
    """
    Test that MappingStimulusFile.num_frames returns the
    expected result
    """
    str_path = str(general_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"mapping_stimulus_file": str_path}
    map_stim = MappingStimulusFile.from_json(dict_repr=dict_repr)
    assert map_stim.num_frames() == general_pkl_fixture['expected_frames']


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
        stim.num_frames()


def test_malformed_replay_pkl(
        behavior_pkl_fixture):
    """
    Test that the correct error is raised when a replay pickle file
    is mal-formed and num_frames is called
    """
    str_path = str(behavior_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"replay_stimulus_file": str_path}
    stim = ReplayStimulusFile.from_json(dict_repr=dict_repr)
    with pytest.raises(RuntimeError,
                       match="When getting num_frames from"):
        stim.num_frames()


def test_malformed_mapping_pkl(
        behavior_pkl_fixture):
    """
    Test that the correct error is raised when a mapping pickle file
    is mal-formed and num_frames is called
    """
    str_path = str(behavior_pkl_fixture['path'].resolve().absolute())
    dict_repr = {"mapping_stimulus_file": str_path}
    stim = MappingStimulusFile.from_json(dict_repr=dict_repr)
    with pytest.raises(RuntimeError,
                       match="When getting num_frames from"):  # noqa W605
        stim.num_frames()
