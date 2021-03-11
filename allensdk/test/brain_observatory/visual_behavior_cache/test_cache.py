import pytest
import json
import hashlib
import pathlib
import boto3
from moto import mock_s3
from allensdk.brain_observatory.visual_behavior_cache.cloud_cache import CloudCache  # noqa: E501
from allensdk.brain_observatory.visual_behavior_cache.file_attributes import CacheFileAttributes  # noqa: E501


@mock_s3
def test_list_all_manifests():
    """
    Test that CloudCache.list_al_manifests() returns the correct result
    """

    test_bucket_name = 'list_manifest_bucket'

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name)

    client = boto3.client('s3', region_name='us-east-1')
    client.put_object(Bucket=test_bucket_name,
                      Key='manifests/manifest_1.json',
                      Body=b'123456')
    client.put_object(Bucket=test_bucket_name,
                      Key='manifests/manifest_2.json',
                      Body=b'123456')
    client.put_object(Bucket=test_bucket_name,
                      Key='junk.txt',
                      Body=b'123456')

    class DummyCache(CloudCache):
        _bucket_name = test_bucket_name

    cache = DummyCache('/my/cache/dir')

    assert cache.manifest_file_names == ['manifest_1.json', 'manifest_2.json']


@mock_s3
def test_list_all_manifests_many():
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
                          Key=f'manifests/manifest_{ii}.json',
                          Body=b'123456')

    client.put_object(Bucket=test_bucket_name,
                      Key='junk.txt',
                      Body=b'123456')

    class DummyCache(CloudCache):
        _bucket_name = test_bucket_name

    cache = DummyCache('/my/cache/dir')

    expected = list([f'manifest_{ii}.json' for ii in range(2000)])
    expected.sort()
    assert cache.manifest_file_names == expected


@mock_s3
def test_loading_manifest():
    """
    Test loading manifests with CloudCache
    """

    test_bucket_name = 'list_manifest_bucket'

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    client = boto3.client('s3', region_name='us-east-1')

    manifest_1 = {'dataset_version': '1',
                  'metadata_files': {'a.csv': {'uri': 'http://www.junk.com',
                                               's3_version': '1111',
                                               'md5_hash': 'abcde'},
                                     'b.csv': {'uri': 'http://silly.com',
                                               's3_version': '2222',
                                               'md5_hash': 'fghijk'}}}

    manifest_2 = {'dataset_version': '2',
                  'metadata_files': {'c.csv': {'uri': 'http://www.absurd.com',
                                               's3_version': '3333',
                                               'md5_hash': 'lmnop'},
                                     'd.csv': {'uri': 'http://nonsense.com',
                                               's3_version': '4444',
                                               'md5_hash': 'qrstuv'}}}

    client.put_object(Bucket=test_bucket_name,
                      Key='manifests/manifest_1.csv',
                      Body=bytes(json.dumps(manifest_1), 'utf-8'))

    client.put_object(Bucket=test_bucket_name,
                      Key='manifests/manifest_2.csv',
                      Body=bytes(json.dumps(manifest_2), 'utf-8'))

    class DummyCache(CloudCache):
        _bucket_name = test_bucket_name

    cache = DummyCache('/my/cache/dir')
    cache.load_manifest('manifest_1.csv')
    assert cache._manifest._data == manifest_1
    assert cache.version == '1'

    cache.load_manifest('manifest_2.csv')
    assert cache._manifest._data == manifest_2
    assert cache.version == '2'

    with pytest.raises(ValueError) as context:
        cache.load_manifest('manifest_3.csv')
    msg = 'is not one of the valid manifest names'
    assert msg in context.value.args[0]


@mock_s3
def test_file_exists(tmpdir):
    """
    Test that cache._file_exists behaves correctly
    """

    data = b'aakderasjklsafetss77123523asf'
    md5sum = hashlib.md5()
    md5sum.update(data)
    true_checksum = md5sum.hexdigest()
    test_file_path = pathlib.Path(tmpdir)/'junk.txt'
    with open(test_file_path, 'wb') as out_file:
        out_file.write(data)

    # need to populate a bucket in order for
    # CloudCache to be instantiated
    test_bucket_name = 'silly_bucket'
    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    class SillyCache(CloudCache):
        _bucket_name = test_bucket_name

    cache = SillyCache('my/cache/dir')

    # should be true
    good_attribute = CacheFileAttributes('http://silly.url.com',
                                         '12345',
                                         true_checksum,
                                         test_file_path)
    assert cache._file_exists(good_attribute)

    # test when checksum is wrong
    bad_attribute = CacheFileAttributes('http://silly.url.com',
                                        '12345',
                                        'probably_not_the_checksum',
                                        test_file_path)
    assert not cache._file_exists(bad_attribute)

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
    Test that CloudCache._download_file behaves as expected
    """

    md5sum = hashlib.md5()
    data = b'11235813kjlssergwesvsdd'
    md5sum.update(data)
    true_checksum = md5sum.hexdigest()

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

    class DownloadTestCache(CloudCache):
        _bucket_name = test_bucket_name

    cache_dir = pathlib.Path(tmpdir) / 'download/test/cache'
    cache = DownloadTestCache(cache_dir)

    expected_path = cache_dir / true_checksum / 'data/data_file.txt'

    uri = f'http://{test_bucket_name}.s3.amazonaws.com/data/data_file.txt'
    good_attributes = CacheFileAttributes(uri,
                                          version_id,
                                          true_checksum,
                                          expected_path)

    assert not expected_path.exists()
    assert cache._download_file(good_attributes)
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum


@mock_s3
def test_download_file_multiple_versions(tmpdir):
    """
    Test that CloudCache._download_file behaves as expected
    when there are multiple versions of the same file in the
    bucket

    (This is really just testing that S3's versioning behaves the
    way we think it does)
    """

    md5sum = hashlib.md5()
    data_1 = b'11235813kjlssergwesvsdd'
    md5sum.update(data_1)
    true_checksum_1 = md5sum.hexdigest()

    md5sum = hashlib.md5()
    data_2 = b'zzzzxxxxyyyywwwwjjjj'
    md5sum.update(data_2)
    true_checksum_2 = md5sum.hexdigest()

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

    class DownloadVersionTestCache(CloudCache):
        _bucket_name = test_bucket_name

    cache_dir = pathlib.Path(tmpdir) / 'download/test/cache'
    cache = DownloadVersionTestCache(cache_dir)

    uri = f'http://{test_bucket_name}.s3.amazonaws.com/data/data_file.txt'

    # download first version of file
    expected_path = cache_dir / true_checksum_1 / 'data/data_file.txt'

    good_attributes = CacheFileAttributes(uri,
                                          version_id_1,
                                          true_checksum_1,
                                          expected_path)

    assert not expected_path.exists()
    assert cache._download_file(good_attributes)
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum_1

    # download second version of file
    expected_path = cache_dir / true_checksum_2 / 'data/data_file.txt'

    good_attributes = CacheFileAttributes(uri,
                                          version_id_2,
                                          true_checksum_2,
                                          expected_path)

    assert not expected_path.exists()
    assert cache._download_file(good_attributes)
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum_2


@mock_s3
def test_re_download_file(tmpdir):
    """
    Test that CloudCache._download_file will re-download a file
    when it has been altered locally
    """

    md5sum = hashlib.md5()
    data = b'11235813kjlssergwesvsdd'
    md5sum.update(data)
    true_checksum = md5sum.hexdigest()

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

    class ReDownloadTestCache(CloudCache):
        _bucket_name = test_bucket_name

    cache_dir = pathlib.Path(tmpdir) / 'download/test/cache'
    cache = ReDownloadTestCache(cache_dir)

    expected_path = cache_dir / true_checksum / 'data/data_file.txt'

    uri = f'http://{test_bucket_name}.s3.amazonaws.com/data/data_file.txt'
    good_attributes = CacheFileAttributes(uri,
                                          version_id,
                                          true_checksum,
                                          expected_path)

    assert not expected_path.exists()
    assert cache._download_file(good_attributes)
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum

    # now, remove the file, and see if it gets re-downloaded
    expected_path.unlink()
    assert not expected_path.exists()

    assert cache._download_file(good_attributes)
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum

    # now, alter the file, and see if it gets re-downloaded
    with open(expected_path, 'wb') as out_file:
        out_file.write(b'778899')
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() != true_checksum

    assert cache._download_file(good_attributes)
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum


@mock_s3
def test_data_path(tmpdir):
    """
    Test that CloudCache.data_path() correctly downloads files from S3
    """

    md5sum = hashlib.md5()
    data = b'11235813kjlssergwesvsdd'
    md5sum.update(data)
    true_checksum = md5sum.hexdigest()

    test_bucket_name = 'bucket_for_data_path'
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
    manifest['dataset_version'] = '1'
    manifest['metadata_files'] = {}
    uri = f'http://{test_bucket_name}.s3.amazonaws.com/data/data_file.txt'
    data_file = {'uri': uri,
                 's3_version': version_id,
                 'md5_hash': true_checksum}

    manifest['data_files'] = {'only_data_file': data_file}

    client.put_object(Bucket=test_bucket_name,
                      Key='manifests/manifest_1.json',
                      Body=bytes(json.dumps(manifest), 'utf-8'))

    class DataPathCache(CloudCache):
        _bucket_name = test_bucket_name

    cache_dir = pathlib.Path(tmpdir) / "data/path/cache"
    cache = DataPathCache(cache_dir)

    cache.load_manifest('manifest_1.json')

    expected_path = cache_dir / true_checksum / 'data/data_file.txt'
    assert not expected_path.exists()

    result_path = cache.data_path('only_data_file')
    assert result_path == expected_path
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum


@mock_s3
def test_metadata_path(tmpdir):
    """
    Test that CloudCache.metadata_path() correctly downloads files from S3
    """

    md5sum = hashlib.md5()
    data = b'11235813kjlssergwesvsdd'
    md5sum.update(data)
    true_checksum = md5sum.hexdigest()

    test_bucket_name = 'bucket_for_metadata_path'
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
    manifest['dataset_version'] = '1'
    uri = f'http://{test_bucket_name}.s3.amazonaws.com/metadata_file.csv'
    metadata_file = {'uri': uri,
                     's3_version': version_id,
                     'md5_hash': true_checksum}

    manifest['metadata_files'] = {'metadata_file.csv': metadata_file}

    client.put_object(Bucket=test_bucket_name,
                      Key='manifests/manifest_1.json',
                      Body=bytes(json.dumps(manifest), 'utf-8'))

    class MetadataPathCache(CloudCache):
        _bucket_name = test_bucket_name

    cache_dir = pathlib.Path(tmpdir) / "metadata/path/cache"
    cache = MetadataPathCache(cache_dir)

    cache.load_manifest('manifest_1.json')

    expected_path = cache_dir / true_checksum / 'metadata_file.csv'
    assert not expected_path.exists()

    result_path = cache.metadata_path('metadata_file.csv')
    assert result_path == expected_path
    assert expected_path.exists()
    md5sum = hashlib.md5()
    with open(expected_path, 'rb') as in_file:
        md5sum.update(in_file.read())
    assert md5sum.hexdigest() == true_checksum
