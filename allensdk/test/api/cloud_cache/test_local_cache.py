import pathlib
from moto import mock_s3
from .utils import create_bucket
from allensdk.api.cloud_cache.cloud_cache import S3CloudCache
from allensdk.api.cloud_cache.cloud_cache import LocalCache


@mock_s3
def test_local_cache_file_access(tmpdir, example_datasets):
    """
    Create a cache; download some, but not all of the files
    with S3CloudCache; verify that we can access the files
    with LocalCache
    """

    bucket_name = 'local_cache_bucket'
    create_bucket(bucket_name, example_datasets)
    cache_dir = pathlib.Path(tmpdir) / 'cache'
    cloud_cache = S3CloudCache(cache_dir, bucket_name, 'project-x')

    cloud_cache.load_manifest('project-x_manifest_v1.0.0.json')
    cloud_cache.download_data('1')
    cloud_cache.download_data('3')

    cloud_cache.load_manifest('project-x_manifest_v3.0.0.json')
    cloud_cache.download_data('2')

    del cloud_cache

    local_cache = LocalCache(cache_dir, 'project-x')

    manifest_set = set(local_cache.manifest_file_names)
    assert manifest_set == {'project-x_manifest_v1.0.0.json',
                            'project-x_manifest_v3.0.0.json'}

    local_cache.load_manifest('project-x_manifest_v1.0.0.json')
    attr = local_cache.data_path('1')
    assert attr['exists']
    attr = local_cache.data_path('2')
    assert not attr['exists']
    attr = local_cache.data_path('3')
    assert attr['exists']

    local_cache.load_manifest('project-x_manifest_v3.0.0.json')
    attr = local_cache.data_path('1')
    assert attr['exists']  # because file 1 is the same in v1.0 and v3.0
    attr = local_cache.data_path('2')
    assert attr['exists']
    attr = local_cache.data_path('3')
    assert not attr['exists']
