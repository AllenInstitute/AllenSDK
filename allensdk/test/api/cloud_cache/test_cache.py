import pytest
import json
import hashlib
import pathlib
import pandas as pd
import io
import boto3
from moto import mock_s3
from .utils import create_bucket
from allensdk.api.cloud_cache.cloud_cache import OutdatedManifestWarning
from allensdk.api.cloud_cache.cloud_cache import S3CloudCache  # noqa: E501
from allensdk.api.cloud_cache.file_attributes import CacheFileAttributes  # noqa: E501


@mock_s3
def test_list_all_manifests(tmpdir):
    """
    Test that S3CloudCache.list_al_manifests() returns the correct result
    """

    test_bucket_name = 'list_manifest_bucket'

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name)

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='proj/manifests/manifest_v1.0.0.json',
                      Body=b'123456')
    client.put_object(Bucket=test_bucket_name,
                      Key='proj/manifests/manifest_v2.0.0.json',
                      Body=b'123456')
    client.put_object(Bucket=test_bucket_name,
                      Key='junk.txt',
                      Body=b'123456')

    cache = S3CloudCache(tmpdir, test_bucket_name, 'proj')

    assert cache.manifest_file_names == ['manifest_v1.0.0.json',
                                         'manifest_v2.0.0.json']


@mock_s3
def test_list_all_manifests_many(tmpdir):
    """
    Test the extreme case when there are more manifests than list_objects_v2
    can return at a time
    """

    test_bucket_name = 'list_manifest_bucket'

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name)

    client = boto3.client('s3', region_name='us-east-1')
    for ii in range(2000):
        client.put_object(Bucket=test_bucket_name,
                          Key=f'proj/manifests/manifest_{ii}.json',
                          Body=b'123456')

    client.put_object(Bucket=test_bucket_name,
                      Key='junk.txt',
                      Body=b'123456')

    cache = S3CloudCache(tmpdir, test_bucket_name, 'proj')

    expected = list([f'manifest_{ii}.json' for ii in range(2000)])
    expected.sort()
    assert cache.manifest_file_names == expected


@mock_s3
def test_loading_manifest(tmpdir):
    """
    Test loading manifests with S3CloudCache
    """

    test_bucket_name = 'list_manifest_bucket'

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    client = boto3.client('s3', region_name='us-east-1')

    manifest_1 = {'manifest_version': '1',
                  'metadata_file_id_column_name': 'file_id',
                  'data_pipeline': 'placeholder',
                  'project_name': 'sam-beckett',
                  'data_files': {},
                  'metadata_files': {'a.csv': {'url': 'http://www.junk.com',
                                               'version_id': '1111',
                                               'file_hash': 'abcde'},
                                     'b.csv': {'url': 'http://silly.com',
                                               'version_id': '2222',
                                               'file_hash': 'fghijk'}}}

    manifest_2 = {'manifest_version': '2',
                  'metadata_file_id_column_name': 'file_id',
                  'data_pipeline': 'placeholder',
                  'project_name': 'al',
                  'data_files': {},
                  'metadata_files': {'c.csv': {'url': 'http://www.absurd.com',
                                               'version_id': '3333',
                                               'file_hash': 'lmnop'},
                                     'd.csv': {'url': 'http://nonsense.com',
                                               'version_id': '4444',
                                               'file_hash': 'qrstuv'}}}

    client.put_object(Bucket=test_bucket_name,
                      Key='proj/manifests/manifest_v1.0.0.json',
                      Body=bytes(json.dumps(manifest_1), 'utf-8'))

    client.put_object(Bucket=test_bucket_name,
                      Key='proj/manifests/manifest_v2.0.0.json',
                      Body=bytes(json.dumps(manifest_2), 'utf-8'))

    cache = S3CloudCache(pathlib.Path(tmpdir), test_bucket_name, 'proj')
    assert cache.current_manifest is None
    cache.load_manifest('manifest_v1.0.0.json')
    assert cache._manifest._data == manifest_1
    assert cache.version == '1'
    assert cache.file_id_column == 'file_id'
    assert cache.metadata_file_names == ['a.csv', 'b.csv']
    assert cache.current_manifest == 'manifest_v1.0.0.json'

    cache.load_manifest('manifest_v2.0.0.json')
    assert cache._manifest._data == manifest_2
    assert cache.version == '2'
    assert cache.file_id_column == 'file_id'
    assert cache.metadata_file_names == ['c.csv', 'd.csv']

    with pytest.raises(ValueError) as context:
        cache.load_manifest('manifest_v3.0.0.json')
    msg = 'is not one of the valid manifest names'
    assert msg in context.value.args[0]


@mock_s3
def test_file_exists(tmpdir):
    """
    Test that cache._file_exists behaves correctly
    """

    data = b'aakderasjklsafetss77123523asf'
    hasher = hashlib.blake2b()
    hasher.update(data)
    true_checksum = hasher.hexdigest()
    test_file_path = pathlib.Path(tmpdir)/'junk.txt'
    with open(test_file_path, 'wb') as out_file:
        out_file.write(data)

    # need to populate a bucket in order for
    # S3CloudCache to be instantiated
    test_bucket_name = 'silly_bucket'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    cache = S3CloudCache(tmpdir, test_bucket_name, 'proj')

    # should be true
    good_attribute = CacheFileAttributes('http://silly.url.com',
                                         '12345',
                                         true_checksum,
                                         test_file_path)
    assert cache._file_exists(good_attribute)

    # test when file path is wrong
    bad_path = pathlib.Path('definitely/not/a/file.txt')
    bad_attribute = CacheFileAttributes('http://silly.url.com',
                                        '12345',
                                        true_checksum,
                                        bad_path)

    assert not cache._file_exists(bad_attribute)

    # test when path exists but is not a file
    bad_attribute = CacheFileAttributes('http://silly.url.com',
                                        '12345',
                                        true_checksum,
                                        pathlib.Path(tmpdir))
    with pytest.raises(RuntimeError) as context:
        cache._file_exists(bad_attribute)
    assert 'but is not a file' in context.value.args[0]


@mock_s3
def test_download_file(tmpdir):
    """
    Test that S3CloudCache._download_file behaves as expected
    """

    hasher = hashlib.blake2b()
    data = b'11235813kjlssergwesvsdd'
    hasher.update(data)
    true_checksum = hasher.hexdigest()

    test_bucket_name = 'bucket_for_download'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # turn on bucket versioning
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='data/data_file.txt',
                      Body=data)

    response = client.list_object_versions(Bucket=test_bucket_name)
    version_id = response['Versions'][0]['VersionId']

    cache_dir = pathlib.Path(tmpdir) / 'download/test/cache'
    cache = S3CloudCache(cache_dir, test_bucket_name, 'proj')

    expected_path = cache_dir / true_checksum / 'data/data_file.txt'

    url = f'http://{test_bucket_name}.s3.amazonaws.com/data/data_file.txt'
    good_attributes = CacheFileAttributes(url,
                                          version_id,
                                          true_checksum,
                                          expected_path)

    assert not expected_path.exists()
    cache._download_file(good_attributes)
    assert expected_path.exists()
    hasher = hashlib.blake2b()
    with open(expected_path, 'rb') as in_file:
        hasher.update(in_file.read())
    assert hasher.hexdigest() == true_checksum


@mock_s3
def test_download_file_multiple_versions(tmpdir):
    """
    Test that S3CloudCache._download_file behaves as expected
    when there are multiple versions of the same file in the
    bucket

    (This is really just testing that S3's versioning behaves the
    way we think it does)
    """

    hasher = hashlib.blake2b()
    data_1 = b'11235813kjlssergwesvsdd'
    hasher.update(data_1)
    true_checksum_1 = hasher.hexdigest()

    hasher = hashlib.blake2b()
    data_2 = b'zzzzxxxxyyyywwwwjjjj'
    hasher.update(data_2)
    true_checksum_2 = hasher.hexdigest()

    assert true_checksum_2 != true_checksum_1

    test_bucket_name = 'bucket_for_download_versions'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # turn on bucket versioning
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='data/data_file.txt',
                      Body=data_1)

    response = client.list_object_versions(Bucket=test_bucket_name)
    version_id_1 = response['Versions'][0]['VersionId']

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='data/data_file.txt',
                      Body=data_2)

    response = client.list_object_versions(Bucket=test_bucket_name)
    version_id_2 = None
    for v in response['Versions']:
        if v['IsLatest']:
            version_id_2 = v['VersionId']
    assert version_id_2 is not None
    assert version_id_2 != version_id_1

    cache_dir = pathlib.Path(tmpdir) / 'download/test/cache'
    cache = S3CloudCache(cache_dir, test_bucket_name, 'proj')

    url = f'http://{test_bucket_name}.s3.amazonaws.com/data/data_file.txt'

    # download first version of file
    expected_path = cache_dir / true_checksum_1 / 'data/data_file.txt'

    good_attributes = CacheFileAttributes(url,
                                          version_id_1,
                                          true_checksum_1,
                                          expected_path)

    assert not expected_path.exists()
    cache._download_file(good_attributes)
    assert expected_path.exists()
    hasher = hashlib.blake2b()
    with open(expected_path, 'rb') as in_file:
        hasher.update(in_file.read())
    assert hasher.hexdigest() == true_checksum_1

    # download second version of file
    expected_path = cache_dir / true_checksum_2 / 'data/data_file.txt'

    good_attributes = CacheFileAttributes(url,
                                          version_id_2,
                                          true_checksum_2,
                                          expected_path)

    assert not expected_path.exists()
    cache._download_file(good_attributes)
    assert expected_path.exists()
    hasher = hashlib.blake2b()
    with open(expected_path, 'rb') as in_file:
        hasher.update(in_file.read())
    assert hasher.hexdigest() == true_checksum_2


@mock_s3
def test_re_download_file(tmpdir):
    """
    Test that S3CloudCache._download_file will re-download a file
    when it has been removed from the local system
    """

    hasher = hashlib.blake2b()
    data = b'11235813kjlssergwesvsdd'
    hasher.update(data)
    true_checksum = hasher.hexdigest()

    test_bucket_name = 'bucket_for_re_download'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # turn on bucket versioning
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='data/data_file.txt',
                      Body=data)

    response = client.list_object_versions(Bucket=test_bucket_name)
    version_id = response['Versions'][0]['VersionId']

    cache_dir = pathlib.Path(tmpdir) / 'download/test/cache'
    cache = S3CloudCache(cache_dir, test_bucket_name, 'proj')

    expected_path = cache_dir / true_checksum / 'data/data_file.txt'

    url = f'http://{test_bucket_name}.s3.amazonaws.com/data/data_file.txt'
    good_attributes = CacheFileAttributes(url,
                                          version_id,
                                          true_checksum,
                                          expected_path)

    assert not expected_path.exists()
    cache._download_file(good_attributes)
    assert expected_path.exists()
    hasher = hashlib.blake2b()
    with open(expected_path, 'rb') as in_file:
        hasher.update(in_file.read())
    assert hasher.hexdigest() == true_checksum

    # now, remove the file, and see if it gets re-downloaded
    expected_path.unlink()
    assert not expected_path.exists()

    cache._download_file(good_attributes)
    assert expected_path.exists()
    hasher = hashlib.blake2b()
    with open(expected_path, 'rb') as in_file:
        hasher.update(in_file.read())
    assert hasher.hexdigest() == true_checksum


@mock_s3
def test_download_data(tmpdir):
    """
    Test that S3CloudCache.download_data() correctly downloads files from S3
    """

    hasher = hashlib.blake2b()
    data = b'11235813kjlssergwesvsdd'
    hasher.update(data)
    true_checksum = hasher.hexdigest()

    test_bucket_name = 'bucket_for_download_data'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # turn on bucket versioning
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='data/data_file.txt',
                      Body=data)

    response = client.list_object_versions(Bucket=test_bucket_name)
    version_id = response['Versions'][0]['VersionId']

    manifest = {}
    manifest['manifest_version'] = '1'
    manifest['project_name'] = "project-z"
    manifest['metadata_file_id_column_name'] = 'file_id'
    manifest['metadata_files'] = {}
    url = f'http://{test_bucket_name}.s3.amazonaws.com/project-z/data/data_file.txt'  # noqa: E501
    data_file = {'url': url,
                 'version_id': version_id,
                 'file_hash': true_checksum}

    manifest['data_files'] = {'only_data_file': data_file}
    manifest['data_pipeline'] = 'placeholder'

    client.put_object(Bucket=test_bucket_name,
                      Key='proj/manifests/manifest_v1.0.0.json',
                      Body=bytes(json.dumps(manifest), 'utf-8'))

    cache_dir = pathlib.Path(tmpdir) / "data/path/cache"
    cache = S3CloudCache(cache_dir, test_bucket_name, 'proj')

    cache.load_manifest('manifest_v1.0.0.json')

    expected_path = cache_dir / 'project-z-1' / 'data/data_file.txt'
    assert not expected_path.exists()

    # test data_path
    attr = cache.data_path('only_data_file')
    assert attr['local_path'] == expected_path
    assert not attr['exists']

    # NOTE: commenting out because moto does not support
    # list_object_versions and this is becoming difficult

    # result_path = cache.download_data('only_data_file')
    # assert result_path == expected_path
    # assert expected_path.exists()
    # hasher = hashlib.blake2b()
    # with open(expected_path, 'rb') as in_file:
    #     hasher.update(in_file.read())
    # assert hasher.hexdigest() == true_checksum

    # test that data_path detects that the file now exists
    # attr = cache.data_path('only_data_file')
    # assert attr['local_path'] == expected_path
    # assert attr['exists']


@mock_s3
def test_download_metadata(tmpdir):
    """
    Test that S3CloudCache.download_metadata() correctly
    downloads files from S3
    """

    hasher = hashlib.blake2b()
    data = b'11235813kjlssergwesvsdd'
    hasher.update(data)
    true_checksum = hasher.hexdigest()

    test_bucket_name = 'bucket_for_download_metadata'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # turn on bucket versioning
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')
    meta_version = client.put_object(Bucket=test_bucket_name,
                                     Key='metadata_file.csv',
                                     Body=data)["VersionId"]

    response = client.list_object_versions(Bucket=test_bucket_name)
    version_id = response['Versions'][0]['VersionId']

    manifest = {}
    manifest['manifest_version'] = '1'
    manifest['project_name'] = "project4"
    manifest['metadata_file_id_column_name'] = 'file_id'
    url = f'http://{test_bucket_name}.s3.amazonaws.com/project4/metadata_file.csv'  # noqa: E501
    metadata_file = {'url': url,
                     'version_id': version_id,
                     'file_hash': true_checksum}

    manifest['metadata_files'] = {'metadata_file.csv': metadata_file}
    manifest['data_files'] = {}
    manifest['data_pipeline'] = 'placeholder'

    client.put_object(Bucket=test_bucket_name,
                      Key='proj/manifests/manifest_v1.0.0.json',
                      Body=bytes(json.dumps(manifest), 'utf-8'))

    cache_dir = pathlib.Path(tmpdir) / "metadata/path/cache"
    cache = S3CloudCache(cache_dir, test_bucket_name, 'proj')

    cache.load_manifest('manifest_v1.0.0.json')

    expected_path = cache_dir / "project4-1" / 'metadata_file.csv'
    assert not expected_path.exists()

    # test that metadata_path also works
    attr = cache.metadata_path('metadata_file.csv')
    assert attr['local_path'] == expected_path
    assert not attr['exists']

    def response_fun(Bucket, Prefix):
        # moto doesn't cover list_object_versions
        return {"Versions": [{
            "VersionId": meta_version,
            "Key": "metadata_file.csv",
            "Size": 12}]}
    # cache.s3_client.list_object_versions = response_fun

    # NOTE: commenting out because moto does not support
    # list_object_versions and this is becoming difficult

    # result_path = cache.download_metadata('metadata_file.csv')
    # assert result_path == expected_path
    # assert expected_path.exists()
    # hasher = hashlib.blake2b()
    # with open(expected_path, 'rb') as in_file:
    #     hasher.update(in_file.read())
    # assert hasher.hexdigest() == true_checksum

    # # test that metadata_path detects that the file now exists
    # attr = cache.metadata_path('metadata_file.csv')
    # assert attr['local_path'] == expected_path
    # assert attr['exists']


@mock_s3
def test_metadata(tmpdir):
    """
    Test that S3CloudCache.metadata() returns the expected pandas DataFrame
    """
    data = {}
    data['mouse_id'] = [1, 4, 6, 8]
    data['sex'] = ['F', 'F', 'M', 'M']
    data['age'] = ['P50', 'P46', 'P23', 'P40']
    true_df = pd.DataFrame(data)

    with io.StringIO() as stream:
        true_df.to_csv(stream, index=False)
        stream.seek(0)
        data = bytes(stream.read(), 'utf-8')

    hasher = hashlib.blake2b()
    hasher.update(data)
    true_checksum = hasher.hexdigest()

    test_bucket_name = 'bucket_for_metadata'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # turn on bucket versioning
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='metadata_file.csv',
                      Body=data)

    response = client.list_object_versions(Bucket=test_bucket_name)
    version_id = response['Versions'][0]['VersionId']

    manifest = {}
    manifest['manifest_version'] = '1'
    manifest['project_name'] = "project-X"
    manifest['metadata_file_id_column_name'] = 'file_id'
    url = f'http://{test_bucket_name}.s3.amazonaws.com/metadata_file.csv'
    metadata_file = {'url': url,
                     'version_id': version_id,
                     'file_hash': true_checksum}

    manifest['metadata_files'] = {'metadata_file.csv': metadata_file}
    manifest['data_files'] = {}
    manifest['data_pipeline'] = 'placeholder'

    client.put_object(Bucket=test_bucket_name,
                      Key='proj/manifests/manifest_v1.0.0.json',
                      Body=bytes(json.dumps(manifest), 'utf-8'))

    cache_dir = pathlib.Path(tmpdir) / "metadata/cache"
    cache = S3CloudCache(cache_dir, test_bucket_name, 'proj')
    cache.load_manifest('manifest_v1.0.0.json')

    metadata_df = cache.get_metadata('metadata_file.csv')
    assert true_df.equals(metadata_df)


@mock_s3
def test_latest_manifest(tmpdir, example_datasets_with_metadata):
    """
    Test that the methods which return the latest and latest downloaded
    manifest file names work correctly
    """
    bucket_name = 'latest_manifest_bucket'
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=example_datasets_with_metadata['metadata'])

    cache_dir = pathlib.Path(tmpdir) / 'cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    assert cache.latest_downloaded_manifest_file == ''

    cache.load_manifest('project-x_manifest_v7.0.0.json')
    cache.load_manifest('project-x_manifest_v3.0.0.json')
    cache.load_manifest('project-x_manifest_v2.0.0.json')

    assert cache.latest_manifest_file == 'project-x_manifest_v15.0.0.json'

    expected = 'project-x_manifest_v7.0.0.json'
    assert cache.latest_downloaded_manifest_file == expected


@mock_s3
def test_outdated_manifest_warning(tmpdir, example_datasets_with_metadata):
    """
    Test that a warning is raised the first time you try to load an outdated
    manifest
    """

    bucket_name = 'outdated_manifest_bucket'
    metadatasets = example_datasets_with_metadata['metadata']
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=metadatasets)

    cache_dir = pathlib.Path(tmpdir) / 'cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    m_warn_type = 'OutdatedManifestWarning'

    with pytest.warns(OutdatedManifestWarning) as warnings:
        cache.load_manifest('project-x_manifest_v7.0.0.json')
    ct = 0
    for w in warnings.list:
        if w._category_name == m_warn_type:
            msg = str(w.message)
            assert 'is not the most up to date' in msg
            assert 'S3CloudCache.compare_manifests' in msg
            assert 'load_latest_manifest' in msg
            ct += 1
    assert ct > 0

    # assert no warning is raised the second time by catching
    # any warnings that are emitted and making sure they are
    # not OutdatedManifestWarnings
    with pytest.warns(None) as warnings:
        cache.load_manifest('project-x_manifest_v11.0.0.json')
    if len(warnings) > 0:
        for w in warnings.list:
            assert w._category_name != 'OutdatedManifestWarning'


@mock_s3
def test_list_all_downloaded(tmpdir, example_datasets_with_metadata):
    """
    Test that list_all_downloaded_manifests works
    """

    bucket_name = 'outdated_manifest_bucket'
    metadatasets = example_datasets_with_metadata['metadata']
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=metadatasets)

    cache_dir = pathlib.Path(tmpdir) / 'cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    assert cache.list_all_downloaded_manifests() == []

    cache.load_manifest('project-x_manifest_v5.0.0.json')
    assert cache.current_manifest == 'project-x_manifest_v5.0.0.json'
    cache.load_manifest('project-x_manifest_v2.0.0.json')
    assert cache.current_manifest == 'project-x_manifest_v2.0.0.json'
    cache.load_manifest('project-x_manifest_v3.0.0.json')
    assert cache.current_manifest == 'project-x_manifest_v3.0.0.json'

    expected = {'project-x_manifest_v5.0.0.json',
                'project-x_manifest_v2.0.0.json',
                'project-x_manifest_v3.0.0.json'}
    downloaded = set(cache.list_all_downloaded_manifests())
    assert downloaded == expected


@mock_s3
def test_latest_manifest_warning(tmpdir, example_datasets_with_metadata):
    """
    Test that the correct warning is emitted when the user tries
    to load_latest_manifest but that has not been downloaded yet
    """

    bucket_name = 'outdated_manifest_bucket'
    metadatasets = example_datasets_with_metadata['metadata']
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=metadatasets)

    cache_dir = pathlib.Path(tmpdir) / 'cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    cache.load_manifest('project-x_manifest_v4.0.0.json')

    with pytest.warns(OutdatedManifestWarning) as warnings:
        cache.load_latest_manifest()
    assert len(warnings) == 1
    msg = str(warnings[0].message)
    assert 'project-x_manifest_v4.0.0.json' in msg
    assert 'project-x_manifest_v15.0.0.json' in msg
    assert 'It is possible that some data files' in msg
    cmd = "S3CloudCache.load_manifest('project-x_manifest_v4.0.0.json')"
    assert cmd in msg


@mock_s3
def test_load_last_manifest(tmpdir, example_datasets_with_metadata):
    """
    Test that load_last_manifest works
    """
    bucket_name = 'load_lst_manifest_bucket'
    metadatasets = example_datasets_with_metadata['metadata']
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=metadatasets)

    cache_dir = pathlib.Path(tmpdir) / 'load_last_cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    # check that load_last_manifest in a new cache loads the
    # latest manifest without emitting a warning
    with pytest.warns(None) as warnings:
        cache.load_last_manifest()
    ct = 0
    for w in warnings.list:
        if w._category_name == 'OutdatedManifestWarning':
            ct += 1
    assert ct == 0
    assert cache.current_manifest == 'project-x_manifest_v15.0.0.json'

    cache.load_manifest('project-x_manifest_v7.0.0.json')

    del cache

    # check that load_last_manifest on an old cache emits the
    # expected warning and loads the correct manifest
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')
    expected = 'A more up to date version of the '
    expected += 'dataset -- project-x_manifest_v15.0.0.json '
    expected += '-- exists online'
    with pytest.warns(OutdatedManifestWarning,
                      match=expected) as warnings:
        cache.load_last_manifest()

    assert cache.current_manifest == 'project-x_manifest_v7.0.0.json'
    cache.load_manifest('project-x_manifest_v4.0.0.json')
    del cache

    # repeat the above test, making sure the correct manifest is
    # loaded again
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')
    expected = 'A more up to date version of the '
    expected += 'dataset -- project-x_manifest_v15.0.0.json '
    expected += '-- exists online'
    with pytest.warns(OutdatedManifestWarning,
                      match=expected) as warnings:
        cache.load_last_manifest()

    assert cache.current_manifest == 'project-x_manifest_v4.0.0.json'


@mock_s3
def test_corrupted_load_last_manifest(tmpdir,
                                      example_datasets_with_metadata):
    """
    Test that load_last_manifest works when the record of the last
    manifest has been corrupted
    """
    bucket_name = 'load_lst_manifest_bucket'
    metadatasets = example_datasets_with_metadata['metadata']
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=metadatasets)

    cache_dir = pathlib.Path(tmpdir) / 'load_last_cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')
    cache.load_manifest('project-x_manifest_v9.0.0.json')
    fname = cache._manifest_last_used.resolve()
    del cache
    with open(fname, 'w') as out_file:
        out_file.write('babababa')
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')
    expected = 'Loading latest version -- project-x_manifest_v15.0.0.json'
    with pytest.warns(UserWarning, match=expected):
        cache.load_last_manifest()
    assert cache.current_manifest == 'project-x_manifest_v15.0.0.json'
