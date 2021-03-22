import pytest
import pandas as pd
from unittest.mock import MagicMock

from allensdk.brain_observatory.behavior.project_apis.data_io import \
        behavior_project_cloud_api as cloudapi


class MockCache():
    def __init__(self,
                 behavior_session_table,
                 ophys_session_table,
                 ophys_experiment_table,
                 cachedir):
        self.file_id_column = "file_id"
        self.session_table_path = cachedir / "session.csv"
        self.behavior_session_table_path = cachedir / "behavior_session.csv"
        self.ophys_experiment_table_path = cachedir / "ophys_experiment.csv"

        ophys_session_table.to_csv(self.session_table_path, index=False)
        behavior_session_table.to_csv(self.behavior_session_table_path,
                                      index=False)
        ophys_experiment_table.to_csv(self.ophys_experiment_table_path,
                                      index=False)

        self._manifest = MagicMock()
        self._manifest.metadata_file_names = ["behavior_session_table",
                                              "ophys_session_table",
                                              "ophys_experiment_table"]

    def download_metadata(self, mname):
        mymap = {
                "behavior_session_table": self.behavior_session_table_path,
                "ophys_session_table": self.session_table_path,
                "ophys_experiment_table": self.ophys_experiment_table_path}
        return mymap[mname]

    def download_data(self, idstr):
        return idstr


@pytest.fixture
def mock_cache(request, tmpdir):
    bst = request.param.get("behavior_session_table")
    ost = request.param.get("ophys_session_table")
    oet = request.param.get("ophys_experiment_table")

    # round-trip the tables through csv to pick up
    # pandas mods to lists
    fname = tmpdir / "my.csv"
    bst.to_csv(fname, index=False)
    bst = pd.read_csv(fname)
    ost.to_csv(fname, index=False)
    ost = pd.read_csv(fname)
    oet.to_csv(fname, index=False)
    oet = pd.read_csv(fname)
    yield (MockCache(bst, ost, oet, tmpdir), request.param)


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
                    "file_id": [4, 5, 6, 7, 8, 9]})},
                ],
        indirect=["mock_cache"])
def test_BehaviorProjectCloudApi(mock_cache, monkeypatch):
    mocked_cache, expected = mock_cache
    api = cloudapi.BehaviorProjectCloudApi(mocked_cache,
                                           skip_version_check=True)

    # behavior session table as expected
    bost = api.get_behavior_only_session_table()
    assert bost.index.name == "behavior_session_id"
    bost = bost.reset_index()
    ebost = expected["behavior_session_table"]
    for k in ["behavior_session_id", "file_id"]:
        pd.testing.assert_series_equal(bost[k], ebost[k])
    for k in ["ophys_experiment_id"]:
        assert all([i == j
                    for i, j in zip(bost[k].values, ebost[k].values)])

    # ophys session table as expected
    ost = api.get_session_table()
    assert ost.index.name == "ophys_session_id"
    ost = ost.reset_index()
    eost = expected["ophys_session_table"]
    for k in ["ophys_session_id"]:
        pd.testing.assert_series_equal(ost[k], eost[k])
    for k in ["ophys_experiment_id"]:
        assert all([i == j
                    for i, j in zip(ost[k].values, eost[k].values)])

    # experiment table as expected
    et = api.get_experiment_table()
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
        "pipeline_versions, sdk_version, lookup, exception, match",
        [
            (
                [{
                    "name": "AllenSDK",
                    "version": "2.9.0"}],
                "2.9.0",
                {"pipeline_versions": {
                    "2.9.0": {"AllenSDK": ["2.9.0", "3.0.0"]}}},
                None,
                ""),
            (
                [{
                    "name": "AllenSDK",
                    "version": "2.9.0"}],
                "2.9.0",
                {"pipeline_versions": {
                    "2.9.0": {"AllenSDK": ["2.9.1", "3.0.0"]}}},
                cloudapi.BehaviorCloudCacheVersionException,
                r".*version be >=2.9.1 and <3.0.0.*"),
            (
                [{
                    "name": "AllenSDK",
                    "version": "2.9.0"}],
                "2.9.0",
                {"pipeline_versions": {
                    "2.9.0": {"AllenSDK": ["2.8.0", "2.9.0"]}}},
                cloudapi.BehaviorCloudCacheVersionException,
                r".*version be >=2.8.0 and <2.9.0.*"),
            (
                [{
                    "name": "AllenSDK",
                    "version": "2.10.0"}],
                "2.9.0",
                {"pipeline_versions": {
                    "2.9.0": {"AllenSDK": ["2.8.0", "2.9.0"]}}},
                cloudapi.BehaviorCloudCacheVersionException,
                r"no version compatibility .*"),
            (
                [{
                    "name": "AllenSDK",
                    "version": "2.10.0"},
                 {
                     "name": "AllenSDK",
                     "version": "2.10.1"}],
                "2.9.0",
                {"pipeline_versions": {
                    "2.9.0": {"AllenSDK": ["2.8.0", "2.9.0"]}}},
                cloudapi.BehaviorCloudCacheVersionException,
                r"expected to find 1 and only 1 .*"),
            ])
def test_compatibility(pipeline_versions, sdk_version, lookup,
                       exception, match):
    if exception is None:
        cloudapi.version_check(pipeline_versions, sdk_version, lookup)
        return
    with pytest.raises(exception, match=match):
        cloudapi.version_check(pipeline_versions, sdk_version, lookup)
