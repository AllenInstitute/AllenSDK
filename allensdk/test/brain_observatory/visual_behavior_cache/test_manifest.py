import pytest
import json
import io
import pathlib
from allensdk.brain_observatory.visual_behavior_cache.manifest import Manifest
from allensdk.brain_observatory.visual_behavior_cache.file_attributes import CacheFileAttributes  # noqa: E501


def test_constructor():
    """
    Make sure that the Manifest class __init__ runs and
    raises an error if you give it an unexpected cache_dir
    """
    _ = Manifest('my/cache/dir')
    _ = Manifest(pathlib.Path('my/other/cache/dir'))
    with pytest.raises(ValueError) as context:
        _ = Manifest(1234.2)
    msg = "cache_dir must be either a str or a pathlib.Path; "
    msg += "got <class 'float'>"
    assert context.value.args[0] == msg


def test_load(tmpdir):
    """
    Bare bones check to verify that Manifest.load can be run and that it
    will raise the correct error when the JSONized manifest is not a dict
    """

    good_manifest = {}
    good_manifest['dataset_version'] = 'A'
    metadata_files = {}
    metadata_files['z.txt'] = []
    metadata_files['x.txt'] = []
    metadata_files['y.txt'] = []
    good_manifest['metadata_files'] = metadata_files

    stream = io.StringIO()
    stream.write(json.dumps(good_manifest))
    stream.seek(0)

    mfest = Manifest(pathlib.Path(tmpdir) / 'my/cache/dir')
    mfest.load(stream)
    assert mfest.version == 'A'
    assert mfest.metadata_file_names == ['x.txt', 'y.txt', 'z.txt']
    assert mfest._cache_dir == pathlib.Path(str(tmpdir)+'/my/cache/dir')

    del stream

    # test that you can load a new manifest.json into the same Manifest
    good_manifest = {}
    good_manifest['dataset_version'] = 'B'
    metadata_files = {}
    metadata_files['n.txt'] = []
    metadata_files['k.txt'] = []
    metadata_files['u.txt'] = []
    good_manifest['metadata_files'] = metadata_files

    stream = io.StringIO()
    stream.write(json.dumps(good_manifest))
    stream.seek(0)

    mfest.load(stream)
    assert mfest.version == 'B'
    assert mfest.metadata_file_names == ['k.txt', 'n.txt', 'u.txt']

    del stream

    # test that an error is raised when manifest.json is not a dict
    bad_manifest = ['a', 'b', 'c']
    stream = io.StringIO()
    stream.write(json.dumps(bad_manifest))
    stream.seek(0)
    with pytest.raises(ValueError) as context:
        mfest.load(stream)
    msg = "Expected to deserialize manifest into a dict; "
    msg += "instead got <class 'list'>"
    assert context.value.args[0] == msg


def test_create_file_attributes():
    """
    Test that Manifest._create_file_attributes correctly
    handles input parameters (this is mostly a test of
    local_path generation)
    """
    mfest = Manifest('/my/cache/dir')
    attr = mfest._create_file_attributes('http://my.url.com/path/to/file.txt',
                                         '12345',
                                         'aaabbbcccddd')

    assert isinstance(attr, CacheFileAttributes)
    assert attr.uri == 'http://my.url.com/path/to/file.txt'
    assert attr.version_id == '12345'
    assert attr.md5_checksum == 'aaabbbcccddd'
    expected_path = '/my/cache/dir/aaabbbcccddd/path/to/file.txt'
    assert attr.local_path == pathlib.Path(expected_path)


def test_metadata_file_attributes():
    """
    Test that Manifest.metadata_file_attributes returns the
    correct CacheFileAttributes object and raises the correct
    error when you ask for a metadata file that does not exist
    """

    manifest = {}
    metadata_files = {}
    metadata_files['a.txt'] = {'uri': 'http://my.url.com/path/to/a.txt',
                               's3_version': '12345',
                               'md5_hash': 'abcde'}
    metadata_files['b.txt'] = {'uri': 'http://my.other.url.com/different/path/to/b.txt',  # noqa: E501
                               's3_version': '67890',
                               'md5_hash': 'fghijk'}

    manifest['metadata_files'] = metadata_files
    manifest['dataset_version'] = '000'

    stream = io.StringIO()
    stream.write(json.dumps(manifest))
    stream.seek(0)

    mfest = Manifest('/my/cache/dir/')
    mfest.load(stream)

    a_obj = mfest.metadata_file_attributes('a.txt')
    assert a_obj.uri == 'http://my.url.com/path/to/a.txt'
    assert a_obj.version_id == '12345'
    assert a_obj.md5_checksum == 'abcde'
    expected = pathlib.Path('/my/cache/dir/abcde/path/to/a.txt')
    assert a_obj.local_path == expected

    b_obj = mfest.metadata_file_attributes('b.txt')
    assert b_obj.uri == 'http://my.other.url.com/different/path/to/b.txt'
    assert b_obj.version_id == '67890'
    assert b_obj.md5_checksum == 'fghijk'
    expected = pathlib.Path('/my/cache/dir/fghijk/different/path/to/b.txt')
    assert b_obj.local_path == expected

    # test that the correct error is raised when you ask
    # for a metadata file that does not exist

    with pytest.raises(ValueError) as context:
        _ = mfest.metadata_file_attributes('c.txt')
    msg = "c.txt\nis not in self.metadata_file_names"
    assert msg in context.value.args[0]


def test_data_file_attributes():
    """
    Test that Manifest.data_file_attributes returns the correct
    CacheFileAttributes object and raises the correct error when
    you ask for a data file that does not exist
    """
    manifest = {}
    manifest['metadata_files'] = {}
    manifest['dataset_version'] = '0'
    data_files = {}
    data_files['a'] = {'uri': 'http://my.url.com/path/to/a.nwb',
                       's3_version': '12345',
                       'md5_hash': 'abcde'}
    data_files['b'] = {'uri': 'http://my.other.url.com/different/path/b.nwb',
                       's3_version': '67890',
                       'md5_hash': 'fghijk'}
    manifest['data_files'] = data_files

    stream = io.StringIO()
    stream.write(json.dumps(manifest))
    stream.seek(0)

    mfest = Manifest('/my/cache/dir')
    mfest.load(stream)

    a_obj = mfest.data_file_attributes('a')
    assert a_obj.uri == 'http://my.url.com/path/to/a.nwb'
    assert a_obj.version_id == '12345'
    assert a_obj.md5_checksum == 'abcde'
    expected = '/my/cache/dir/abcde/path/to/a.nwb'
    assert a_obj.local_path == pathlib.Path(expected)

    b_obj = mfest.data_file_attributes('b')
    assert b_obj.uri == 'http://my.other.url.com/different/path/b.nwb'
    assert b_obj.version_id == '67890'
    assert b_obj.md5_checksum == 'fghijk'
    expected = '/my/cache/dir/fghijk/different/path/b.nwb'
    assert b_obj.local_path == pathlib.Path(expected)

    with pytest.raises(ValueError) as context:
        _ = mfest.data_file_attributes('c')
    msg = "file_id: c\nIs not a data file listed in manifest:"
    assert msg in context.value.args[0]


def test_loading_two_manifests():
    """
    Test that Manifest behaves correctly after re-running load() on
    a different manifest
    """

    # create two manifests, meant to represents different versions
    # of the same dataset

    manifest_1 = {}
    metadata_1 = {}
    metadata_1['metadata_a.csv'] = {'uri': 'http://aaa.com/path/to/a.csv',
                                    's3_version': '12345',
                                    'md5_hash': 'abcde'}
    metadata_1['metadata_b.csv'] = {'uri': 'http://bbb.com/other/path/b.csv',
                                    's3_version': '67890',
                                    'md5_hash': 'fghijk'}
    manifest_1['metadata_files'] = metadata_1
    data_1 = {}
    data_1['c'] = {'uri': 'http://ccc.com/third/path/c.csv',
                   's3_version': '11121',
                   'md5_hash': 'lmnopq'}
    data_1['d'] = {'uri': 'http://ddd.com/fourth/path/d.csv',
                   's3_version': '31415',
                   'md5_hash': 'rstuvw'}

    manifest_1['data_files'] = data_1
    manifest_1['dataset_version'] = '1'

    stream_1 = io.StringIO()
    stream_1.write(json.dumps(manifest_1))
    stream_1.seek(0)

    manifest_2 = {}
    metadata_2 = {}
    metadata_2['metadata_a.csv'] = {'uri': 'http://aaa.com/path/to/a.csv',
                                    's3_version': '161718',
                                    'md5_hash': 'xyzab'}
    metadata_2['metadata_f.csv'] = {'uri': 'http://fff.com/fifth/path/f.csv',
                                    's3_version': '192021',
                                    'md5_hash': 'cdefghi'}
    manifest_2['metadata_files'] = metadata_2
    data_2 = {}
    data_2['c'] = {'uri': 'http://ccc.com/third/path/c.csv',
                   's3_version': '222324',
                   'md5_hash': 'jklmnop'}
    data_2['g'] = {'uri': 'http://ggg.com/sixth/path/g.csv',
                   's3_version': '25262728',
                   'md5_hash': 'qrstuvwxy'}

    manifest_2['data_files'] = data_2
    manifest_2['dataset_version'] = '2'

    stream_2 = io.StringIO()
    stream_2.write(json.dumps(manifest_2))
    stream_2.seek(0)

    mfest = Manifest('/my/cache/dir')

    # load the first version of the manifest and check results

    mfest.load(stream_1)
    assert mfest.version == '1'
    assert mfest.metadata_file_names == ['metadata_a.csv', 'metadata_b.csv']

    m_obj = mfest.metadata_file_attributes('metadata_a.csv')
    assert m_obj.uri == 'http://aaa.com/path/to/a.csv'
    assert m_obj.version_id == '12345'
    assert m_obj.md5_checksum == 'abcde'
    expected = '/my/cache/dir/abcde/path/to/a.csv'
    assert m_obj.local_path == pathlib.Path(expected)

    m_obj = mfest.metadata_file_attributes('metadata_b.csv')
    assert m_obj.uri == 'http://bbb.com/other/path/b.csv'
    assert m_obj.version_id == '67890'
    assert m_obj.md5_checksum == 'fghijk'
    expected = '/my/cache/dir/fghijk/other/path/b.csv'
    assert m_obj.local_path == pathlib.Path(expected)

    d_obj = mfest.data_file_attributes('c')
    assert d_obj.uri == 'http://ccc.com/third/path/c.csv'
    assert d_obj.version_id == '11121'
    assert d_obj.md5_checksum == 'lmnopq'
    expected = '/my/cache/dir/lmnopq/third/path/c.csv'
    assert d_obj.local_path == pathlib.Path(expected)

    d_obj = mfest.data_file_attributes('d')
    assert d_obj.uri == 'http://ddd.com/fourth/path/d.csv'
    assert d_obj.version_id == '31415'
    assert d_obj.md5_checksum == 'rstuvw'
    expected = '/my/cache/dir/rstuvw/fourth/path/d.csv'
    assert d_obj.local_path == pathlib.Path(expected)

    # now load the second manifest and make sure that everything
    # changes accordingly

    mfest.load(stream_2)
    assert mfest.version == '2'
    assert mfest.metadata_file_names == ['metadata_a.csv', 'metadata_f.csv']

    m_obj = mfest.metadata_file_attributes('metadata_a.csv')
    assert m_obj.uri == 'http://aaa.com/path/to/a.csv'
    assert m_obj.version_id == '161718'
    assert m_obj.md5_checksum == 'xyzab'
    expected = '/my/cache/dir/xyzab/path/to/a.csv'
    assert m_obj.local_path == pathlib.Path(expected)

    m_obj = mfest.metadata_file_attributes('metadata_f.csv')
    assert m_obj.uri == 'http://fff.com/fifth/path/f.csv'
    assert m_obj.version_id == '192021'
    assert m_obj.md5_checksum == 'cdefghi'
    expected = '/my/cache/dir/cdefghi/fifth/path/f.csv'
    assert m_obj.local_path == pathlib.Path(expected)

    with pytest.raises(ValueError):
        _ = mfest.metadata_file_attributes('metadata_b.csv')

    d_obj = mfest.data_file_attributes('c')
    assert d_obj.uri == 'http://ccc.com/third/path/c.csv'
    assert d_obj.version_id == '222324'
    assert d_obj.md5_checksum == 'jklmnop'
    expected = '/my/cache/dir/jklmnop/third/path/c.csv'
    assert d_obj.local_path == pathlib.Path(expected)

    d_obj = mfest.data_file_attributes('g')
    assert d_obj.uri == 'http://ggg.com/sixth/path/g.csv'
    assert d_obj.version_id == '25262728'
    assert d_obj.md5_checksum == 'qrstuvwxy'
    expected = '/my/cache/dir/qrstuvwxy/sixth/path/g.csv'
    assert d_obj.local_path == pathlib.Path(expected)

    with pytest.raises(ValueError):
        _ = mfest.data_file_attributes('d')
