import pathlib
from moto import mock_s3
from .utils import create_bucket
from allensdk.api.cloud_cache.cloud_cache import S3CloudCache


@mock_s3
def test_summarize_comparison(tmpdir, example_datasets_with_metadata):
    """
    Test that CloudCacheBase.summarize_comparison reports the correct
    changes when comparing two manifests
    """
    bucket_name = 'summarizing_bucket'
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=example_datasets_with_metadata['metadata'])

    cache_dir = pathlib.Path(tmpdir) / 'cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v2.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 1
    assert ('data/f2.txt', 'data/f2.txt deleted') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v3.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 1
    assert ('data/f2.txt',
            'data/f2.txt renamed data/f4.txt') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v4.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 1
    assert ('data/f3.txt', 'data/f3.txt changed') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v5.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 1
    assert ('data/f4.txt', 'data/f4.txt created') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v6.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 2
    assert ('data/f2.txt', 'data/f2.txt deleted') in log['data_changes']
    assert ('data/f1.txt', 'data/f1.txt changed') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v7.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 2
    assert ('data/f2.txt', 'data/f2.txt deleted') in log['data_changes']
    assert ('data/f3.txt', 'data/f3.txt '
            'renamed data/f5.txt') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v8.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 2
    assert ('data/f2.txt', 'data/f2.txt deleted') in log['data_changes']
    assert ('data/f5.txt', 'data/f5.txt created') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v9.0.0.json')

    assert len(log['metadata_changes']) == 0
    assert len(log['data_changes']) == 2
    assert ('data/f3.txt', 'data/f3.txt renamed '
            'data/f4.txt') in log['data_changes']
    assert ('data/f5.txt', 'data/f5.txt created') in log['data_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v10.0.0.json')

    assert len(log['data_changes']) == 0
    assert len(log['metadata_changes']) == 1
    assert ('project_metadata/metadata_2.csv',
            'project_metadata/metadata_2.csv '
            'deleted') in log['metadata_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v11.0.0.json')

    assert len(log['data_changes']) == 0
    assert len(log['metadata_changes']) == 1
    assert ('project_metadata/metadata_2.csv',
            'project_metadata/metadata_2.csv renamed '
            'project_metadata/metadata_4.csv') in log['metadata_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v12.0.0.json')

    assert len(log['data_changes']) == 0
    assert len(log['metadata_changes']) == 1
    assert ('project_metadata/metadata_3.csv',
            'project_metadata/metadata_3.csv '
            'changed') in log['metadata_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v13.0.0.json')

    assert len(log['data_changes']) == 0
    assert len(log['metadata_changes']) == 1
    assert ('project_metadata/metadata_4.csv',
            'project_metadata/metadata_4.csv '
            'created') in log['metadata_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v14.0.0.json')
    assert len(log['data_changes']) == 1
    assert len(log['metadata_changes']) == 1
    assert ('data/f2.txt', 'data/f2.txt deleted') in log['data_changes']
    assert ('project_metadata/metadata_3.csv',
            'project_metadata/metadata_3.csv renamed '
            'project_metadata/metadata_4.csv') in log['metadata_changes']

    log = cache.summarize_comparison('project-x_manifest_v1.0.0.json',
                                     'project-x_manifest_v15.0.0.json')
    assert len(log['data_changes']) == 3
    assert len(log['metadata_changes']) == 3

    ans1 = ('data/f1.txt', 'data/f1.txt renamed data/f4.txt')
    ans2 = ('data/f5.txt', 'data/f5.txt created')
    ans3 = ('data/f6.txt', 'data/f6.txt created')

    assert set(log['data_changes']) == {ans1, ans2, ans3}

    ans1 = ('project_metadata/metadata_2.csv',
            'project_metadata/metadata_2.csv renamed '
            'project_metadata/metadata_4.csv')
    ans2 = ('project_metadata/metadata_1.csv',
            'project_metadata/metadata_1.csv deleted')
    ans3 = ('project_metadata/metadata_3.csv',
            'project_metadata/metadata_3.csv deleted')

    assert set(log['metadata_changes']) == {ans1, ans2, ans3}


@mock_s3
@mock_s3
def test_compare_manifesst_string(tmpdir, example_datasets_with_metadata):
    """
    Test that CloudCacheBase.compare_manifests reports the correct
    changes when comparing two manifests
    """
    bucket_name = 'compare_manifest_bucket'
    create_bucket(bucket_name,
                  example_datasets_with_metadata['data'],
                  metadatasets=example_datasets_with_metadata['metadata'])

    cache_dir = pathlib.Path(tmpdir) / 'cache'
    cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    msg = cache.compare_manifests('project-x_manifest_v1.0.0.json',
                                  'project-x_manifest_v15.0.0.json')

    expected = 'Changes going from\n'
    expected += 'project-x_manifest_v1.0.0.json\n'
    expected += 'to\n'
    expected += 'project-x_manifest_v15.0.0.json\n\n'
    expected += 'project_metadata/metadata_1.csv deleted\n'
    expected += 'project_metadata/metadata_2.csv renamed '
    expected += 'project_metadata/metadata_4.csv\n'
    expected += 'project_metadata/metadata_3.csv deleted\n'
    expected += 'data/f1.txt renamed data/f4.txt\n'
    expected += 'data/f5.txt created\n'
    expected += 'data/f6.txt created\n'

    assert msg == expected
