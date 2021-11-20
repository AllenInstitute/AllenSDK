import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

from allensdk.api.cloud_cache.manifest import Manifest
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io import behavior_project_cloud_api as cloudapi  # noqa: E501
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.\
        data_io.behavior_project_cloud_api import MANIFEST_COMPATIBILITY


class MockCache():
    def __init__(self,
                 behavior_session_table,
                 ophys_session_table,
                 ophys_experiment_table,
                 ophys_cells_table,
                 cachedir):
        self.file_id_column = "file_id"
        self.session_table_path = cachedir / "session.csv"
        self.behavior_session_table_path = cachedir / "behavior_session.csv"
        self.ophys_experiment_table_path = cachedir / "ophys_experiment.csv"
        self.ophys_cells_table_path = cachedir / "ophys_cells.csv"

        ophys_session_table.to_csv(self.session_table_path, index=False)
        ophys_cells_table.to_csv(self.ophys_cells_table_path, index=False)
        behavior_session_table.to_csv(self.behavior_session_table_path,
                                      index=False)
        ophys_experiment_table.to_csv(self.ophys_experiment_table_path,
                                      index=False)

        self._manifest = MagicMock()
        self._manifest.metadata_file_names = ["behavior_session_table",
                                              "ophys_session_table",
                                              "ophys_experiment_table",
                                              "ophys_cells_table"]
        self._metadata_name_path_map = {
                "behavior_session_table": self.behavior_session_table_path,
                "ophys_session_table": self.session_table_path,
                "ophys_experiment_table": self.ophys_experiment_table_path,
                "ophys_cells_table": self.ophys_cells_table_path}

    def download_metadata(self, fname):
        return self._metadata_name_path_map[fname]

    def download_data(self, file_id):
        return file_id

    def metadata_path(self, fname):
        local_path = self._metadata_name_path_map[fname]
        return {
            'local_path': local_path,
            'exists': Path(local_path).exists()
        }

    def data_path(self, file_id):
        return {
            'local_path': file_id,
            'exists': True
        }

    def load_last_manifest(self):
        return None


@pytest.fixture
def mock_cache(request, tmpdir):
    bst = request.param.get("behavior_session_table")
    ost = request.param.get("ophys_session_table")
    oet = request.param.get("ophys_experiment_table")
    clt = request.param.get("ophys_cells_table")

    # round-trip the tables through csv to pick up
    # pandas mods to lists
    fname = tmpdir / "my.csv"
    bst.to_csv(fname, index=False)
    bst = pd.read_csv(fname)
    ost.to_csv(fname, index=False)
    ost = pd.read_csv(fname)
    oet.to_csv(fname, index=False)
    oet = pd.read_csv(fname)
    clt.to_csv(fname, index=False)
    clt = pd.read_csv(fname)
    yield (MockCache(bst, ost, oet, clt, tmpdir), request.param)


@pytest.mark.parametrize(
        "mock_cache",
        [
            {
                "behavior_session_table": pd.DataFrame({
                    "behavior_session_id": [1, 2, 3, 4],
                    "ophys_experiment_id": [4, 5, 6, [7, 8, 9]],
                    "file_id": [4, 5, 6, None]}),
                "ophys_session_table": pd.DataFrame({
                    "ophys_session_id": [10, 11, 12, 13],
                    "ophys_experiment_id": [4, 5, 6, [7, 8, 9]]}),
                "ophys_experiment_table": pd.DataFrame({
                    "ophys_experiment_id": [4, 5, 6, 7, 8, 9],
                    "file_id": [4, 5, 6, 7, 8, 9]}),
                "ophys_cells_table": pd.DataFrame({
                    "cell_roi_id": [4, 5, 6],
                    "cell_specimen_id": [104, 105, 106],
                    "ophys_experiment_id": [4, 5, 6]})}
                ],
        indirect=["mock_cache"])
@pytest.mark.parametrize("local", [True, False])
def test_BehaviorProjectCloudApi(mock_cache, monkeypatch, local):
    mocked_cache, expected = mock_cache
    api = cloudapi.BehaviorProjectCloudApi(mocked_cache,
                                           skip_version_check=True,
                                           local=False)
    if local:
        api = cloudapi.BehaviorProjectCloudApi(mocked_cache,
                                               skip_version_check=True,
                                               local=True)

    # behavior session table as expected
    bost = api.get_behavior_session_table()
    assert bost.index.name == "behavior_session_id"
    bost = bost.reset_index()
    ebost = expected["behavior_session_table"]
    for k in ["behavior_session_id", "file_id"]:
        pd.testing.assert_series_equal(bost[k], ebost[k])
    for k in ["ophys_experiment_id"]:
        assert all([i == j
                    for i, j in zip(bost[k].values, ebost[k].values)])

    # ophys session table as expected
    ost = api.get_ophys_session_table()
    assert ost.index.name == "ophys_session_id"
    ost = ost.reset_index()
    eost = expected["ophys_session_table"]
    for k in ["ophys_session_id"]:
        pd.testing.assert_series_equal(ost[k], eost[k])
    for k in ["ophys_experiment_id"]:
        assert all([i == j
                    for i, j in zip(ost[k].values, eost[k].values)])

    # experiment table as expected
    et = api.get_ophys_experiment_table()
    assert et.index.name == "ophys_experiment_id"
    et = et.reset_index()
    pd.testing.assert_frame_equal(et, expected["ophys_experiment_table"])

    # get_behavior_session returns expected value
    # both directly and via experiment table
    def mock_nwb(nwb_path):
        return nwb_path
    monkeypatch.setattr(cloudapi.BehaviorSession, "from_nwb_path", mock_nwb)
    assert api.get_behavior_session(2) == "5"
    assert api.get_behavior_session(4) == "7"

    # direct check only for ophys experiment
    monkeypatch.setattr(cloudapi.BehaviorOphysExperiment,
                        "from_nwb_path", mock_nwb)
    assert api.get_behavior_ophys_experiment(8) == "8"


@pytest.mark.parametrize(
        "manifest_version, data_pipeline_version, cmin, cmax, exception",
        [
            ("0.0.1", "2.9.0", "0.0.0", "1.0.0", False),
            ("1.0.1", "2.9.0", "0.0.0", "1.0.0", True)
            ])
def test_version_check(manifest_version, data_pipeline_version,
                       cmin, cmax, exception):
    if exception:
        with pytest.raises(cloudapi.BehaviorCloudCacheVersionException,
                           match=f".*{data_pipeline_version}"):
            cloudapi.version_check(manifest_version, data_pipeline_version,
                                   cmin, cmax)
    else:
        cloudapi.version_check(manifest_version, data_pipeline_version,
                               cmin, cmax)


def test_from_local_cache(monkeypatch):
    mock_manifest = create_autospec(Manifest)
    mock_manifest.metadata_file_names = {
        'ophys_experiment_table',
        'ophys_session_table',
        'behavior_session_table',
        'ophys_cells_table'
    }
    mock_manifest._data_pipeline = [
        {
            "name": "AllenSDK",
            "version": "2.11.0",
            "comment": "This is a test entry. NOT REAL."
        }
    ]
    mock_manifest.version = MANIFEST_COMPATIBILITY[0]

    mock_local_cache = create_autospec(cloudapi.LocalCache)
    type(mock_local_cache.return_value)._manifest = mock_manifest

    mock_static_local_cache = create_autospec(cloudapi.StaticLocalCache)
    type(mock_static_local_cache.return_value)._manifest = mock_manifest

    with monkeypatch.context() as m:
        m.setattr(cloudapi, "LocalCache", mock_local_cache)
        m.setattr(cloudapi, "StaticLocalCache", mock_static_local_cache)

        # Test from_local_cache with use_static_cache=False
        try:
            cloudapi.BehaviorProjectCloudApi.from_local_cache(
                "first_cache_dir", "project_1", "ui_1", use_static_cache=False
            )
        # Because cache is a mock, the following calls in the load_manifest
        # method of BehaviorProjectCloudApi will fail with TypeError:
        # self._get_ophys_session_table()
        # self._get_behavior_session_table()
        # self._get_ophys_experiment_table()
        except (TypeError, FileNotFoundError):
            pass

        mock_local_cache.assert_called_once_with(
            "first_cache_dir", "project_1", "ui_1"
        )

        # Test from_local_cache with use_static_cache=True
        try:
            cloudapi.BehaviorProjectCloudApi.from_local_cache(
                "second_cache_dir", "project_2", "ui_2", use_static_cache=True
            )
        # Because cache is a mock, the following calls in the load_manifest
        # method of BehaviorProjectCloudApi will fail with TypeError:
        # self._get_ophys_session_table()
        # self._get_behavior_session_table()
        # self._get_ophys_experiment_table()
        except (TypeError, FileNotFoundError):
            pass

        mock_static_local_cache.assert_called_once_with(
            "second_cache_dir", "project_2", "ui_2"
        )
