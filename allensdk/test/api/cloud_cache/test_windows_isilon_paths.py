import re
import json
from pathlib import Path

from allensdk.api.cloud_cache.cloud_cache import CloudCacheBase
from allensdk.api.cloud_cache.manifest import Manifest


def test_windows_path_to_isilon(monkeypatch, tmpdir):
    """
    This test is just meant to verify on Windows CI instances
    that, if a path to the `/allen/` shared file store is used as
    cache_dir, the path to files will come out useful (i.e. without any
    spurious C:/ prepended as in AllenSDK issue #1964
    """

    cache_dir = Path(tmpdir)

    manifest_1 = {'manifest_version': '1',
                  'metadata_file_id_column_name': 'file_id',
                  'data_pipeline': 'placeholder',
                  'project_name': 'my-project',
                  'metadata_files': {'a.csv': {'url': 'http://www.junk.com/path/to/a.csv',  # noqa: E501
                                               'version_id': '1111',
                                               'file_hash': 'abcde'},
                                     'b.csv': {'url': 'http://silly.com/path/to/b.csv',  # noqa: E501
                                               'version_id': '2222',
                                               'file_hash': 'fghijk'}},
                  'data_files': {'data_1': {'url': 'http://www.junk.com/data/path/data.csv',  # noqa: E501
                                            'version_id': '1111',
                                            'file_hash': 'lmnopqrst'}}
                  }
    manifest_path = tmpdir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_1, f)

    def dummy_file_exists(self, m):
        return True

    # we do not want paths to `/allen` to be resolved to
    # a local drive on the user's machine
    bad_windows_pattern = re.compile('^[A-Z]\:')  # noqa: W605

    # make sure pattern is correctly formulated
    m = bad_windows_pattern.search('C:\\a\windows\path')  # noqa: W605
    assert m is not None

    with monkeypatch.context() as ctx:
        class TestCloudCache(CloudCacheBase):

            def _download_file(self, m, o):
                pass

            def _download_manifest(self, m, o):
                pass

            def _list_all_manifests(self):
                pass

        ctx.setattr(TestCloudCache,
                    '_file_exists',
                    dummy_file_exists)

        cache = TestCloudCache(cache_dir, 'proj')
        cache._manifest = Manifest(cache_dir, json_input=manifest_path)

        m_path = cache.metadata_path('a.csv')
        assert bad_windows_pattern.match(str(m_path)) is None
        d_path = cache.data_path('data_1')
        assert bad_windows_pattern.match(str(d_path)) is None
