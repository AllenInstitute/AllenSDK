from typing import Tuple
from pathlib import Path
import pickle
from unittest.mock import create_autospec

import pytest

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import StimulusFile


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
    stimulus_file = StimulusFile.from_json(input_json_dict)
    assert stimulus_file.data == stim_pkl_data

    # Now test caching by deleting the stimulus_file
    stim_pkl_path.unlink()
    stimulus_file_cached = StimulusFile.from_json(input_json_dict)
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
    stimulus_file = StimulusFile.from_lims(mock_db_conn, behavior_session_id)
    assert stimulus_file.data == stim_pkl_data

    # Now test caching by deleting stimulus_file and also asserting db
    # `fetchone` called only once
    stim_pkl_path.unlink()
    stimfile_cached = StimulusFile.from_lims(mock_db_conn, behavior_session_id)
    assert stimfile_cached.data == stim_pkl_data

    # This query string has strict formatting requirements
    # in order to pass Mock assert_called_once_with() so don't change it!
    query = f"""
            SELECT
                wkf.storage_directory || wkf.filename AS stim_file
            FROM
                well_known_files wkf
            WHERE
                wkf.attachable_id = {behavior_session_id}
                AND wkf.attachable_type = 'BehaviorSession'
                AND wkf.well_known_file_type_id IN (
                    SELECT id
                    FROM well_known_file_types
                    WHERE name = 'StimulusPickle');
        """

    mock_db_conn.fetchone.assert_called_once_with(query, strict=True)


@pytest.mark.parametrize("stimulus_file_fixture", [
    ({"filename": "test_stim_file_1.pkl"}),
    ({"filename": "mock_stim_pkl_2.pkl"})
], indirect=["stimulus_file_fixture"])
def test_stimulus_file_to_json(stimulus_file_fixture):
    stim_pkl_path, stim_pkl_data = stimulus_file_fixture

    stimulus_file = StimulusFile(filepath=stim_pkl_path)
    obt_json = stimulus_file.to_json()
    assert obt_json == {"behavior_stimulus_file": str(stim_pkl_path)}
