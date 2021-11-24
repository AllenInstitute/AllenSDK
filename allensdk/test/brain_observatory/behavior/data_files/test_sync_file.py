from typing import Tuple
from pathlib import Path
import h5py
from unittest.mock import create_autospec
import numpy as np

import pytest

from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.data_files import SyncFile
from allensdk.brain_observatory.behavior.data_files.sync_file import (
    SYNC_FILE_QUERY_TEMPLATE
)


@pytest.fixture
def sync_file_fixture(request, tmp_path) -> Tuple[Path, dict]:
    default_sync_data = [1, 2, 3, 4, 5]
    sync_data = request.param.get("sync_data", default_sync_data)
    sync_filename = request.param.get("filename", "test_sync_file.h5")

    sync_path = tmp_path / sync_filename
    with h5py.File(sync_path, "w") as f:
        f.create_dataset("data", data=sync_data)

    return (sync_path, sync_data)


def mock_get_sync_data(sync_path):
    with h5py.File(sync_path, "r") as f:
        data = f["data"][:]
    return data


@pytest.mark.parametrize("sync_file_fixture", [
    ({"sync_data": [2, 3, 4, 5]}),
], indirect=["sync_file_fixture"])
def test_sync_file_from_json(monkeypatch, sync_file_fixture):
    sync_path, sync_data = sync_file_fixture

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_files"
            ".sync_file.get_sync_data",
            mock_get_sync_data
        )

        # Basic test case
        input_json_dict = {"sync_file": str(sync_path)}
        sync_file = SyncFile.from_json(input_json_dict)
        assert np.allclose(sync_file.data, sync_data)

        # Now test caching by deleting the sync_file
        sync_path.unlink()
        sync_file_cached = SyncFile.from_json(input_json_dict)
        assert np.allclose(sync_file_cached.data, sync_data)


@pytest.mark.parametrize("sync_file_fixture, ophys_experiment_id", [
    ({"sync_data": [2, 3, 4, 5]}, 12),
    ({"sync_data": [2, 3, 4, 5]}, 8)
], indirect=["sync_file_fixture"])
def test_sync_file_from_lims(
    monkeypatch,
    sync_file_fixture,
    ophys_experiment_id
):
    sync_path, sync_data = sync_file_fixture

    mock_db_conn = create_autospec(PostgresQueryMixin, instance=True)

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_files"
            ".sync_file.get_sync_data",
            mock_get_sync_data
        )

        # Basic test case
        mock_db_conn.fetchone.return_value = str(sync_path)
        sync_file = SyncFile.from_lims(mock_db_conn, ophys_experiment_id)
        np.allclose(sync_file.data, sync_data)

        # Now test caching by deleting sync_file and also asserting db
        # `fetchone` called only once
        sync_path.unlink()
        stimfile_cached = SyncFile.from_lims(mock_db_conn, ophys_experiment_id)
        np.allclose(stimfile_cached.data, sync_data)

    query = SYNC_FILE_QUERY_TEMPLATE.format(
        ophys_experiment_id=ophys_experiment_id
    )

    mock_db_conn.fetchone.assert_called_once_with(query, strict=True)


@pytest.mark.parametrize("sync_file_fixture", [
    ({"filename": "test_sync_file_1.h5"}),
    ({"filename": "mock_sync_file_2.h5"})
], indirect=["sync_file_fixture"])
def test_sync_file_to_json(monkeypatch, sync_file_fixture):
    sync_path, sync_data = sync_file_fixture

    with monkeypatch.context() as m:
        m.setattr(
            "allensdk.brain_observatory.behavior.data_files"
            ".sync_file.get_sync_data",
            mock_get_sync_data
        )
        sync_file = SyncFile(filepath=sync_path)
        obt_json = sync_file.to_json()
        assert obt_json == {"sync_file": str(sync_path)}
