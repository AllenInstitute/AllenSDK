import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

from allensdk.api.cloud_cache.manifest import Manifest
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.\
    data_io import behavior_neuropixels_project_cloud_api as cloudapi  # noqa: E501

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.\
    data_io import project_cloud_api_base as cloudapibase  # noqa: E501

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    utils import version_check, BehaviorCloudCacheVersionException
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession


class MockCache():
    def __init__(self,
                 behavior_session_table,
                 ecephys_session_table,
                 probe_table,
                 channel_table,
                 unit_table,
                 cachedir):
        self.file_id_column = "file_id"
        self.session_table_path = cachedir / "ecephys_sessions.csv"
        self.behavior_session_table_path = cachedir / "behavior_sessions.csv"
        self.unit_table_path = cachedir / "units.csv"
        self.probe_table_path = cachedir / "probes.csv"
        self.channel_table_path = cachedir / "channel.csv"

        ecephys_session_table.to_csv(self.session_table_path, index=False)
        channel_table.to_csv(self.channel_table_path, index=False)
        probe_table.to_csv(self.probe_table_path, index=False)
        unit_table.to_csv(self.unit_table_path, index=False)
        behavior_session_table.to_csv(self.behavior_session_table_path,
                                      index=False)

        self._manifest = MagicMock()
        self._manifest.metadata_file_names = ["behavior_sessions",
                                              "ecephys_session",
                                              "probes",
                                              "channels",
                                              "units"]
        self._metadata_name_path_map = {
                "behavior_sessions": self.behavior_session_table_path,
                "ecephys_sessions": self.session_table_path,
                "channels": self.channel_table_path,
                "probes": self.probe_table_path,
                "units": self.unit_table_path}

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
def mock_cache(tmpdir):
    c = {
            "behavior_sessions": pd.DataFrame({
                "behavior_session_id": [1, 2, 3, 4],
                "ecephys_session_id": [10, 11, 12, 13],
                "mouse_id": [4, 4, 2, 1]}),
            "ecephys_sessions": pd.DataFrame({
                "ecephys_session_id": [10, 11, 12, 13],
                "behavior_session_id": [1, 2, 3, 4],
                "file_id": [10, 11, 12, 13]}),
            "probes": pd.DataFrame({
                "ecephys_probe_id": [4, 5, 6, 7],
                "name": ["probeA", "probeB", "probeA", "probeB"],
                "ecephys_session_id": [10, 10, 11, 11],
                "has_lfp_data": [True] * 4,
                "file_id": [0, 1, 2, 3]
            }),
            "channels": pd.DataFrame({
                "ecephys_channel_id": [14, 15, 16],
                "ecephys_probe_id": [4, 4, 4],
                "ecephys_session_id": [10, 10, 10]}),
            "units": pd.DataFrame({
                "unit_id": [204, 205, 206],
                "ecephys_channel_id": [14, 15, 16],
                "ecephys_probe_id": [4, 4, 4],
                "ecephys_session_id": [10, 10, 10]}),
    }

    # round-trip the tables through csv to pick up
    # pandas mods to lists
    fname = tmpdir / "my.csv"
    c['behavior_sessions'].to_csv(fname, index=False)
    bst = pd.read_csv(fname)

    c['ecephys_sessions'].to_csv(fname, index=False)
    est = pd.read_csv(fname)

    c['probes'].to_csv(fname, index=False)
    pt = pd.read_csv(fname)

    c['channels'].to_csv(fname, index=False)
    ct = pd.read_csv(fname)

    c['units'].to_csv(fname, index=False)
    ut = pd.read_csv(fname)

    yield (MockCache(bst, est, pt, ct, ut, tmpdir), c)


@pytest.mark.parametrize("local", [True, False])
def test_VisualBehaviorNeuropixelsProjectCloudApi(
    mock_cache,
    monkeypatch,
    local
):

    mocked_cache, expected = mock_cache
    api = cloudapi.VisualBehaviorNeuropixelsProjectCloudApi(
        mocked_cache,
        skip_version_check=True,
        local=False)

    if local:
        api = cloudapi.VisualBehaviorNeuropixelsProjectCloudApi(
            mocked_cache,
            skip_version_check=True,
            local=True)

    # behavior session table as expected
    bst = api.get_behavior_session_table()
    assert bst.index.name == "behavior_session_id"
    bst = bst.reset_index()
    bst_expected = expected["behavior_sessions"]
    for k in ["behavior_session_id", "mouse_id"]:
        pd.testing.assert_series_equal(bst[k], bst_expected[k])

    # ecephys session table as expected
    est = api.get_ecephys_session_table()
    assert est.index.name == "ecephys_session_id"
    est = est.reset_index()
    est_expected = expected["ecephys_sessions"]
    for k in ["ecephys_session_id"]:
        pd.testing.assert_series_equal(est[k], est_expected[k])

    # probes table as expected
    pt = api.get_probe_table()
    assert pt.index.name == "ecephys_probe_id"
    pt = pt.reset_index()
    pd.testing.assert_frame_equal(pt, expected["probes"])

    # channels table as expected
    ct = api.get_channel_table()
    assert ct.index.name == "ecephys_channel_id"
    ct = ct.reset_index()
    pd.testing.assert_frame_equal(ct, expected["channels"])

    # units table as expected
    ut = api.get_unit_table()
    assert ut.index.name == "unit_id"
    ut = ut.reset_index()
    pd.testing.assert_frame_equal(ut, expected["units"])

    def mock_nwb(nwb_path, probe_data_path_map):
        return nwb_path

    monkeypatch.setattr(cloudapi.BehaviorEcephysSession,
                        "from_nwb_path", mock_nwb)
    assert api.get_ecephys_session(12) == "12"


@pytest.mark.parametrize('has_lfp_data', [True, False])
def test_probe_data_path_map(mock_cache, monkeypatch, has_lfp_data):
    """tests that probe_data_path_map is set correctly"""
    mocked_cache, _ = mock_cache

    probe_meta = pd.read_csv(mocked_cache.probe_table_path)
    probe_meta['has_lfp_data'] = has_lfp_data
    probe_meta.to_csv(mocked_cache.probe_table_path, index=False)

    api = cloudapi.VisualBehaviorNeuropixelsProjectCloudApi(
        mocked_cache,
        skip_version_check=True,
        local=False)

    def mock_from_nwb_path(nwb_path, probe_data_path_map):
        return probe_data_path_map

    ecephys_session_id = 10
    with patch.object(BehaviorEcephysSession, 'from_nwb_path',
                      wraps=mock_from_nwb_path):
        probe_data_path_map = api.get_ecephys_session(ecephys_session_id)

    probe_meta = api.get_probe_table()
    probe_meta = probe_meta.loc[
        probe_meta['ecephys_session_id'] == ecephys_session_id]

    if has_lfp_data:
        for probe_name in probe_meta['name'].unique():
            obtained_probe_nwb_path = probe_data_path_map[probe_name]()
            expected_probe_nwb_path = str(probe_meta.loc[
                (probe_meta['name'] == probe_name)
            ].iloc[0]['file_id'])
            assert obtained_probe_nwb_path == expected_probe_nwb_path
    else:
        assert probe_data_path_map is None


@pytest.mark.parametrize(
        "manifest_version, data_pipeline_version, cmin, cmax, exception",
        [
            ("0.0.1", "2.9.0", "0.0.0", "1.0.0", False),
            ("1.0.1", "2.9.0", "0.0.0", "1.0.0", True)
            ])
def test_version_check(manifest_version, data_pipeline_version,
                       cmin, cmax, exception):
    if exception:
        with pytest.raises(BehaviorCloudCacheVersionException,
                           match=f".*{data_pipeline_version}"):
            version_check(
                manifest_version,
                data_pipeline_version,
                cmin, cmax)
    else:
        version_check(manifest_version, data_pipeline_version, cmin, cmax)


def test_from_local_cache(monkeypatch):
    mock_manifest = create_autospec(Manifest)
    mock_manifest.metadata_file_names = {
        'ecephys_sessions',
        'behavior_sessions',
        'probes',
        'units',
        'channels'
    }
    mock_manifest._data_pipeline = [
        {
            "name": "AllenSDK",
            "version": "2.11.0",
            "comment": "This is a test entry. NOT REAL."
        }
    ]
    mock_manifest.version = cloudapi \
        .VisualBehaviorNeuropixelsProjectCloudApi.MANIFEST_COMPATIBILITY[0]

    mock_local_cache = create_autospec(cloudapibase.LocalCache)
    type(mock_local_cache.return_value)._manifest = mock_manifest
    mock_static_local_cache = create_autospec(cloudapibase.StaticLocalCache)
    type(mock_static_local_cache.return_value)._manifest = mock_manifest

    with monkeypatch.context() as m:

        m.setattr(cloudapibase, "LocalCache", mock_local_cache)
        m.setattr(cloudapibase, "StaticLocalCache", mock_static_local_cache)

        # Test from_local_cache with use_static_cache=False
        try:
            cloudapi.VisualBehaviorNeuropixelsProjectCloudApi.from_local_cache(
                "first_cache_dir", "project_1", "ui_1", use_static_cache=False
            )
        except (TypeError, FileNotFoundError):
            pass

        mock_local_cache.assert_called_once_with(
            "first_cache_dir", "project_1", "ui_1"
        )

        # Test from_local_cache with use_static_cache=True
        try:
            cloudapi.VisualBehaviorNeuropixelsProjectCloudApi.from_local_cache(
                "second_cache_dir", "project_2", "ui_2", use_static_cache=True
            )
        except (TypeError, FileNotFoundError):
            pass

        mock_static_local_cache.assert_called_once_with(
            "second_cache_dir", "project_2", "ui_2"
        )
