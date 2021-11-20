import pytest
import json
import pathlib
from allensdk.internal.core.lims_utilities import safe_system_path
from allensdk.api.cloud_cache.manifest import Manifest
from allensdk.api.cloud_cache.file_attributes import CacheFileAttributes  # noqa: E501


@pytest.fixture
def meta_json_path(tmpdir):
    jpath = tmpdir / "somejson.json"
    d = {
            "project_name": "X",
            "manifest_version": "Y",
            "metadata_file_id_column_name": "Z",
            "data_pipeline": "ZA",
            "metadata_files": ["ZB", "ZC", "ZD"],
            "data_files": {"AB": "ab", "BC": "bc", "CD": "cd"}}
    with open(jpath, "w") as f:
        json.dump(d, f)
    yield jpath


def test_constructor(meta_json_path):
    """
    Make sure that the Manifest class __init__ runs and
    raises an error if you give it an unexpected cache_dir
    """
    Manifest('my/cache/dir', meta_json_path)
    Manifest(pathlib.Path('my/other/cache/dir'), meta_json_path)
    with pytest.raises(ValueError, match=r"cache_dir must be either a str.*"):
        Manifest(1234.2, meta_json_path)


def test_create_file_attributes(meta_json_path):
    """
    Test that Manifest._create_file_attributes correctly
    handles input parameters (this is mostly a test of
    local_path generation)
    """
    mfest = Manifest('/my/cache/dir', meta_json_path)
    attr = mfest._create_file_attributes('http://my.url.com/path/to/file.txt',
                                         '12345',
                                         'aaabbbcccddd')

    assert isinstance(attr, CacheFileAttributes)
    assert attr.url == 'http://my.url.com/path/to/file.txt'
    assert attr.version_id == '12345'
    assert attr.file_hash == 'aaabbbcccddd'
    expected_path = '/my/cache/dir/X-Y/to/file.txt'
    assert attr.local_path == pathlib.Path(expected_path).resolve()


@pytest.fixture
def manifest_for_metadata(tmpdir):
    jpath = tmpdir / "a_manifest.json"
    manifest = {}
    metadata_files = {}
    metadata_files['a.txt'] = {'url': 'http://my.url.com/path/to/a.txt',
                               'version_id': '12345',
                               'file_hash': 'abcde'}
    metadata_files['b.txt'] = {'url': 'http://my.other.url.com/different/path/to/b.txt',  # noqa: E501
                               'version_id': '67890',
                               'file_hash': 'fghijk'}

    manifest['metadata_files'] = metadata_files
    manifest['data_files'] = {}
    manifest['project_name'] = "some-project"
    manifest['manifest_version'] = '000'
    manifest['metadata_file_id_column_name'] = 'file_id'
    manifest['data_pipeline'] = 'placeholder'
    with open(jpath, "w") as f:
        json.dump(manifest, f)
    yield jpath


def test_metadata_file_attributes(manifest_for_metadata):
    """
    Test that Manifest.metadata_file_attributes returns the
    correct CacheFileAttributes object and raises the correct
    error when you ask for a metadata file that does not exist
    """

    mfest = Manifest('/my/cache/dir/', manifest_for_metadata)

    a_obj = mfest.metadata_file_attributes('a.txt')
    assert a_obj.url == 'http://my.url.com/path/to/a.txt'
    assert a_obj.version_id == '12345'
    assert a_obj.file_hash == 'abcde'
    expected = safe_system_path('/my/cache/dir/some-project-000/to/a.txt')
    expected = pathlib.Path(expected).resolve()
    assert a_obj.local_path == expected

    b_obj = mfest.metadata_file_attributes('b.txt')
    assert b_obj.url == 'http://my.other.url.com/different/path/to/b.txt'
    assert b_obj.version_id == '67890'
    assert b_obj.file_hash == 'fghijk'
    expected = safe_system_path('/my/cache/dir/some-project-000/path/to/b.txt')
    expected = pathlib.Path(expected).resolve()
    assert b_obj.local_path == expected

    # test that the correct error is raised when you ask
    # for a metadata file that does not exist

    with pytest.raises(ValueError) as context:
        _ = mfest.metadata_file_attributes('c.txt')
    msg = "c.txt\nis not in self.metadata_file_names"
    assert msg in context.value.args[0]


@pytest.fixture
def manifest_with_data(tmpdir):
    jpath = tmpdir / "manifest_with files.json"
    manifest = {}
    manifest['metadata_files'] = {}
    manifest['manifest_version'] = '0'
    manifest['project_name'] = "myproject"
    manifest['metadata_file_id_column_name'] = 'file_id'
    manifest['data_pipeline'] = 'placeholder'
    data_files = {}
    data_files['a'] = {'url': 'http://my.url.com/myproject/path/to/a.nwb',
                       'version_id': '12345',
                       'file_hash': 'abcde'}
    data_files['b'] = {'url': 'http://my.other.url.com/different/path/b.nwb',
                       'version_id': '67890',
                       'file_hash': 'fghijk'}
    manifest['data_files'] = data_files
    with open(jpath, "w") as f:
        json.dump(manifest, f)
    yield jpath


def test_data_file_attributes(manifest_with_data):
    """
    Test that Manifest.data_file_attributes returns the correct
    CacheFileAttributes object and raises the correct error when
    you ask for a data file that does not exist
    """
    mfest = Manifest('/my/cache/dir', manifest_with_data)

    a_obj = mfest.data_file_attributes('a')
    assert a_obj.url == 'http://my.url.com/myproject/path/to/a.nwb'
    assert a_obj.version_id == '12345'
    assert a_obj.file_hash == 'abcde'
    expected = safe_system_path('/my/cache/dir/myproject-0/path/to/a.nwb')
    assert a_obj.local_path == pathlib.Path(expected).resolve()

    b_obj = mfest.data_file_attributes('b')
    assert b_obj.url == 'http://my.other.url.com/different/path/b.nwb'
    assert b_obj.version_id == '67890'
    assert b_obj.file_hash == 'fghijk'
    expected = safe_system_path('/my/cache/dir/myproject-0/path/b.nwb')
    assert b_obj.local_path == pathlib.Path(expected).resolve()

    with pytest.raises(ValueError) as context:
        _ = mfest.data_file_attributes('c')
    msg = "file_id: c\nIs not a data file listed in manifest:"
    assert msg in context.value.args[0]


def test_file_attribute_errors(meta_json_path):
    """
    Test that Manifest raises the correct error if you try to get file
    attributes before loading a manifest.json
    """
    mfest = Manifest("/my/cache/dir", meta_json_path)
    with pytest.raises(ValueError,
                       match=r".* not in self.metadata_file_names"):
        mfest.metadata_file_attributes('some_file.txt')

    with pytest.raises(ValueError,
                       match=r".* not a data file listed in manifest"):
        mfest.data_file_attributes('other_file.txt')
