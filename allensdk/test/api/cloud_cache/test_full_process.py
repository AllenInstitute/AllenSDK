import pytest
import json
import pathlib
import hashlib
import pandas as pd
import io
import boto3
from moto import mock_s3
from allensdk.api.cloud_cache.cloud_cache import S3CloudCache


@mock_s3
def test_full_cache_system(tmpdir):
    """
    Test the process of loading different versions of the same dataset,
    each of which involve different versions of files
    """

    test_bucket_name = 'full_cache_bucket'

    conn = boto3.resource('s3', region_name='us-east-1')
    conn.create_bucket(Bucket=test_bucket_name, ACL='public-read')

    # turn on bucket versioning
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#bucketversioning
    bucket_versioning = conn.BucketVersioning(test_bucket_name)
    bucket_versioning.enable()

    s3_client = boto3.client('s3', region_name='us-east-1')

    # generate data and expected hashes

    true_hashes = {}
    version_id_lookup = {}

    data1_v1 = b'12345678'
    data1_v2 = b'45678901'
    data2_v1 = b'abcdefghijk'
    data2_v2 = b'lmnopqrstuv'
    data3_v1 = b'jklmnopqrst'

    metadata1_v1 = pd.DataFrame({'mouse': [1, 2, 3],
                                 'sex': ['F', 'F', 'M']})

    metadata2_v1 = pd.DataFrame({'experiment': [5, 6, 7],
                                 'file_id': ['data1', 'data2', 'data3']})

    metadata1_v2 = pd.DataFrame({'mouse': [8, 9, 0],
                                 'sex': ['M', 'F', 'M']})

    v1_hashes = {}
    for data, key in zip((data1_v1, data2_v1, data3_v1),
                         ('data1', 'data2', 'data3')):

        hasher = hashlib.blake2b()
        hasher.update(data)
        v1_hashes[key] = hasher.hexdigest()
        s3_client.put_object(Bucket=test_bucket_name,
                             Key=f'proj/data/{key}',
                             Body=data)

    for df, key in zip((metadata1_v1, metadata2_v1),
                       ('proj/metadata1.csv', 'proj/metadata2.csv')):

        with io.StringIO() as stream:
            df.to_csv(stream, index=False)
            stream.seek(0)
            data = bytes(stream.read(), 'utf-8')

        hasher = hashlib.blake2b()
        hasher.update(data)
        v1_hashes[key.replace('proj/', '')] = hasher.hexdigest()
        s3_client.put_object(Bucket=test_bucket_name,
                             Key=key,
                             Body=data)

    true_hashes['v1'] = v1_hashes
    v1_version_id = {}
    response = s3_client.list_object_versions(Bucket=test_bucket_name)
    for v in response['Versions']:
        vkey = v['Key'].replace('proj/', '').replace('data/', '')
        v1_version_id[vkey] = v['VersionId']

    version_id_lookup['v1'] = v1_version_id

    v2_hashes = {}
    v2_version_id = {}
    for data, key in zip((data1_v2, data2_v2),
                         ('data1', 'data2')):

        hasher = hashlib.blake2b()
        hasher.update(data)
        v2_hashes[key] = hasher.hexdigest()
        s3_client.put_object(Bucket=test_bucket_name,
                             Key=f'proj/data/{key}',
                             Body=data)

    s3_client.delete_object(Bucket=test_bucket_name,
                            Key='proj/data/data3')

    with io.StringIO() as stream:
        metadata1_v2.to_csv(stream, index=False)
        stream.seek(0)
        data = bytes(stream.read(), 'utf-8')

    hasher = hashlib.blake2b()
    hasher.update(data)
    v2_hashes['metadata1.csv'] = hasher.hexdigest()
    s3_client.put_object(Bucket=test_bucket_name,
                         Key='proj/metadata1.csv',
                         Body=data)

    s3_client.delete_object(Bucket=test_bucket_name,
                            Key='proj/metadata2.csv')

    true_hashes['v2'] = v2_hashes
    v2_version_id = {}
    response = s3_client.list_object_versions(Bucket=test_bucket_name)
    for v in response['Versions']:
        if not v['IsLatest']:
            continue
        vkey = v['Key'].replace('proj/', '').replace('data/', '')
        v2_version_id[vkey] = v['VersionId']
    version_id_lookup['v2'] = v2_version_id

    # check thata data3 and metadata2.csv do not occur in v2 of
    # the dataset, but other data/metadata files do

    assert 'data3' in version_id_lookup['v1']
    assert 'data3' not in version_id_lookup['v2']
    assert 'data1' in version_id_lookup['v1']
    assert 'data2' in version_id_lookup['v1']
    assert 'data1' in version_id_lookup['v2']
    assert 'data2' in version_id_lookup['v2']
    assert 'metadata1.csv' in version_id_lookup['v1']
    assert 'metadata2.csv' in version_id_lookup['v1']
    assert 'metadata1.csv' in version_id_lookup['v2']
    assert 'metadata2.csv' not in version_id_lookup['v2']

    # build manifests

    manifest_1 = {}
    manifest_1['manifest_version'] = 'A'
    manifest_1['project_name'] = "project-A1"
    manifest_1['metadata_file_id_column_name'] = 'file_id'
    manifest_1['data_pipeline'] = 'placeholder'
    data_files_1 = {}
    for k in ('data1', 'data2', 'data3'):
        obj = {}
        obj['url'] = f'http://{test_bucket_name}.s3.amazonaws.com/proj/data/{k}'  # noqa: E501
        obj['file_hash'] = true_hashes['v1'][k]
        obj['version_id'] = version_id_lookup['v1'][k]
        data_files_1[k] = obj
    manifest_1['data_files'] = data_files_1
    metadata_files_1 = {}
    for k in ('metadata1.csv', 'metadata2.csv'):
        obj = {}
        obj['url'] = f'http://{test_bucket_name}.s3.amazonaws.com/proj/{k}'
        obj['file_hash'] = true_hashes['v1'][k]
        obj['version_id'] = version_id_lookup['v1'][k]
        metadata_files_1[k] = obj
    manifest_1['metadata_files'] = metadata_files_1

    manifest_2 = {}
    manifest_2['manifest_version'] = 'B'
    manifest_2['project_name'] = "project-B2"
    manifest_2['metadata_file_id_column_name'] = 'file_id'
    manifest_2['data_pipeline'] = 'placeholder'
    data_files_2 = {}
    for k in ('data1', 'data2'):
        obj = {}
        obj['url'] = f'http://{test_bucket_name}.s3.amazonaws.com/proj/data/{k}'  # noqa: E501
        obj['file_hash'] = true_hashes['v2'][k]
        obj['version_id'] = version_id_lookup['v2'][k]
        data_files_2[k] = obj
    manifest_2['data_files'] = data_files_2
    metadata_files_2 = {}
    for k in ['metadata1.csv']:
        obj = {}
        obj['url'] = f'http://{test_bucket_name}.s3.amazonaws.com/proj/{k}'
        obj['file_hash'] = true_hashes['v2'][k]
        obj['version_id'] = version_id_lookup['v2'][k]
        metadata_files_2[k] = obj
    manifest_2['metadata_files'] = metadata_files_2

    s3_client.put_object(Bucket=test_bucket_name,
                         Key='proj/manifests/manifest_v1.0.0.json',
                         Body=bytes(json.dumps(manifest_1), 'utf-8'))

    s3_client.put_object(Bucket=test_bucket_name,
                         Key='proj/manifests/manifest_v2.0.0.json',
                         Body=bytes(json.dumps(manifest_2), 'utf-8'))

    # Use S3CloudCache to interact with dataset
    cache_dir = pathlib.Path(tmpdir) / 'my/test/cache'
    cache = S3CloudCache(cache_dir, test_bucket_name, 'proj')

    # load the first version of the dataset

    cache.load_manifest('manifest_v1.0.0.json')
    assert cache.version == 'A'

    # check that metadata dataframes have expected contents
    m1 = cache.get_metadata('metadata1.csv')
    assert metadata1_v1.equals(m1)
    m2 = cache.get_metadata('metadata2.csv')
    assert metadata2_v1.equals(m2)

    # check that data files have expected hashes
    for k in ('data1', 'data2', 'data3'):

        attr = cache.data_path(k)
        assert not attr['exists']

        local_path = cache.download_data(k)
        assert local_path.exists()
        hasher = hashlib.blake2b()
        with open(local_path, 'rb') as in_file:
            hasher.update(in_file.read())
        assert hasher.hexdigest() == true_hashes['v1'][k]

        attr = cache.data_path(k)
        assert attr['exists']

    # now load the second version of the dataset

    cache.load_manifest('manifest_v2.0.0.json')
    assert cache.version == 'B'

    # metadata2.csv should not exist in this version of the dataset
    with pytest.raises(ValueError) as context:
        cache.get_metadata('metadata2.csv')
    assert 'is not in self.metadata_file_names' in context.value.args[0]

    # check that metadata1 has expected contents
    m1 = cache.get_metadata('metadata1.csv')
    assert metadata1_v2.equals(m1)

    # data3 should not exist in this version of the dataset
    with pytest.raises(ValueError) as context:
        _ = cache.download_data('data3')
    assert 'not a data file listed' in context.value.args[0]

    with pytest.raises(ValueError) as context:
        _ = cache.data_path('data3')
    assert 'not a data file listed' in context.value.args[0]

    # check that data1, data2 have expected hashes
    for k in ('data1', 'data2'):
        attr = cache.data_path(k)
        assert not attr['exists']

        local_path = cache.download_data(k)
        assert local_path.exists()
        hasher = hashlib.blake2b()
        with open(local_path, 'rb') as in_file:
            hasher.update(in_file.read())
        assert hasher.hexdigest() == true_hashes['v2'][k]

        attr = cache.data_path(k)
        assert attr['exists']
