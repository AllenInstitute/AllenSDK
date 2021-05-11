import pytest
from .utils import create_bucket, load_dataset
import boto3
from moto import mock_s3
import pathlib
import json

from allensdk.api.cloud_cache.cloud_cache import MissingLocalManifestWarning
from allensdk.api.cloud_cache.cloud_cache import OutdatedManifestWarning
from allensdk.brain_observatory.\
    behavior.behavior_project_cache.behavior_project_cache \
    import VisualBehaviorOphysProjectCache


@mock_s3
def test_manifest_methods(tmpdir, s3_cloud_cache_data):

    cache_dir = pathlib.Path(tmpdir) / "test_manifest_list"
    bucket_name = "vis-behav-test-bucket"
    project_name = "vis-behav-test-proj"
    create_bucket(bucket_name,
                  project_name,
                  s3_cloud_cache_data['data'],
                  s3_cloud_cache_data['metadata'])

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                          bucket_name,
                                                          project_name)

    m_list = cache.list_manifest_file_names()

    v1_name = f'{project_name}_manifest_v0.1.0.json'
    v2_name = f'{project_name}_manifest_v0.2.0.json'

    assert len(m_list) == 2
    assert v1_name in m_list
    assert v2_name in m_list

    cache.load_manifest(v1_name)

    # because the BehaviorProjectCloudApi automatically
    # loads the latest manifest, so the latest manifest
    # will always be the latest_downloaded_manifest
    assert cache.latest_downloaded_manifest_file() == v2_name
    assert cache.latest_manifest_file() == v2_name

    change_msg = cache.compare_manifests(v1_name, v2_name)

    for mname in ('behavior_session_table',
                  'ophys_session_table',
                  'ophys_experiment_table'):
        assert f'project_metadata/{mname} changed' in change_msg

    assert 'ophys_file_1.nwb changed' in change_msg
    assert 'ophys_file_5.nwb created' in change_msg
    assert 'ophys_file_2.nwb' not in change_msg
    assert 'behavior_file_3.nwb' not in change_msg
    assert 'behavior_file_4.nwb' not in change_msg


@mock_s3
def test_local_cache_construction(tmpdir, s3_cloud_cache_data):

    cache_dir = pathlib.Path(tmpdir) / "test_construction"
    bucket_name = "vis-behav-test-bucket"
    project_name = "vis-behav-test-proj"
    create_bucket(bucket_name,
                  project_name,
                  s3_cloud_cache_data['data'],
                  s3_cloud_cache_data['metadata'])

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                          bucket_name,
                                                          project_name)

    v1_name = f'{project_name}_manifest_v0.1.0.json'
    cache.load_manifest(v1_name)
    cache.get_behavior_ophys_experiment(ophys_experiment_id=5111)
    assert cache.fetch_api.cache._downloaded_data_path.is_file()
    cache.fetch_api.cache._downloaded_data_path.unlink()
    assert not cache.fetch_api.cache._downloaded_data_path.is_file()
    del cache

    with pytest.warns(MissingLocalManifestWarning) as warnings:
        cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                              bucket_name,
                                                              project_name)

    cmd = 'VisualBehaviorOphysProjectCache.construct_local_manifest()'
    assert cmd in f'{warnings[0].message}'

    # check that downloaded data is not in local manifest
    # before running construction function (because
    # VisualBehaviorOphysProjectCache automatically
    # downloads metadata files, those will already
    # be in there)
    manifest_path = cache.fetch_api.cache._downloaded_data_path
    with open(manifest_path, 'rb') as in_file:
        local_manifest = json.load(in_file)
    fnames = set([pathlib.Path(k).name for k in local_manifest])
    assert 'ophys_file_1.nwb' not in fnames
    assert len(local_manifest) == 3

    cache.construct_local_manifest()
    assert cache.fetch_api.cache._downloaded_data_path.is_file()

    with open(manifest_path, 'rb') as in_file:
        local_manifest = json.load(in_file)
    fnames = set([pathlib.Path(k).name for k in local_manifest])
    assert 'ophys_file_1.nwb' in fnames
    assert len(local_manifest) == 7  # 6 metadata files and 1 data file


@mock_s3
def test_load_out_of_date_manifest(tmpdir, s3_cloud_cache_data):
    """
    Test that VisualBehaviorOphysProjectCache can load a
    manifest other than the latest and download files
    from that manifest.
    """
    cache_dir = pathlib.Path(tmpdir) / "test_linkage"
    bucket_name = "vis-behav-test-bucket"
    project_name = "vis-behav-test-proj"
    create_bucket(bucket_name,
                  project_name,
                  s3_cloud_cache_data['data'],
                  s3_cloud_cache_data['metadata'])

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                          bucket_name,
                                                          project_name)

    v1_name = f'{project_name}_manifest_v0.1.0.json'
    cache.load_manifest(v1_name)
    for sess_id in (333, 444):
        cache.get_behavior_session(behavior_session_id=sess_id)
    for exp_id in (5111, 5222):
        cache.get_behavior_ophys_experiment(ophys_experiment_id=exp_id)

    v1_dir = cache_dir / f'{project_name}-0.1.0/data'

    # Check that all expected file were downloaded
    dir_glob = v1_dir.glob('*')
    file_names = set()
    file_contents = {}
    for p in dir_glob:
        file_names.add(p.name)
        with open(p, 'rb') as in_file:
            data = in_file.read()
        file_contents[p.name] = data
    expected = {'ophys_file_1.nwb', 'ophys_file_2.nwb',
                'behavior_file_3.nwb', 'behavior_file_4.nwb'}

    assert file_names == expected

    expected = {}
    expected['ophys_file_1.nwb'] = b'abcde'
    expected['ophys_file_2.nwb'] = b'fghijk'
    expected['behavior_file_3.nwb'] = b'12345'
    expected['behavior_file_4.nwb'] = b'67890'

    assert file_contents == expected


@mock_s3
@pytest.mark.parametrize("delete_cache", [True, False])
def test_file_linkage(tmpdir, s3_cloud_cache_data, delete_cache):
    """
    Test that symlinks are used where appropriate

    if delete_cache == True, will delete the local cache
    file between loading v1 and v2 manifests, then run
    construct_local_cache() to make sure that the symlinks
    are still properly constructed
    """
    cache_dir = pathlib.Path(tmpdir) / "test_linkage"
    bucket_name = "vis-behav-test-bucket"
    project_name = "vis-behav-test-proj"
    create_bucket(bucket_name,
                  project_name,
                  s3_cloud_cache_data['data'],
                  s3_cloud_cache_data['metadata'])

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                          bucket_name,
                                                          project_name)

    v1_name = f'{project_name}_manifest_v0.1.0.json'
    v2_name = f'{project_name}_manifest_v0.2.0.json'
    v1_dir = cache_dir / f'{project_name}-0.1.0/data'
    v2_dir = cache_dir / f'{project_name}-0.2.0/data'

    assert cache.current_manifest() == v2_name
    assert cache.list_all_downloaded_manifests() == [v2_name]

    cache.load_manifest(v1_name)
    assert cache.current_manifest() == v1_name
    assert cache.list_all_downloaded_manifests() == [v1_name,
                                                     v2_name]

    for sess_id in (333, 444):
        cache.get_behavior_session(behavior_session_id=sess_id)
    for exp_id in (5111, 5222):
        cache.get_behavior_ophys_experiment(ophys_experiment_id=exp_id)

    v1_glob = v1_dir.glob('*')
    v1_paths = {}
    for p in v1_glob:
        v1_paths[p.name] = p

    if delete_cache:
        local_cache = cache.fetch_api.cache._downloaded_data_path
        assert local_cache.is_file()
        local_cache.unlink()
        assert not local_cache.is_file()
        del cache

        cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                              bucket_name,
                                                              project_name)
        cache.construct_local_manifest()

    cache.load_manifest(v2_name)
    assert cache.current_manifest() == v2_name
    for sess_id in (777, 888):
        cache.get_behavior_session(behavior_session_id=sess_id)
    for exp_id in (5444, 5666, 5777):
        cache.get_behavior_ophys_experiment(ophys_experiment_id=exp_id)

    v2_glob = v2_dir.glob('*')
    v2_paths = {}
    for p in v2_glob:
        v2_paths[p.name] = p

    # check symlinks
    for name in ('ophys_file_2.nwb',
                 'behavior_file_3.nwb',
                 'behavior_file_4.nwb'):

        assert v2_paths[name].is_symlink()
        assert v2_paths[name].resolve() == v1_paths[name].resolve()
        assert v2_paths[name].absolute() != v1_paths[name].absolute()

    name = 'ophys_file_1.nwb'
    assert not v2_paths[name].is_symlink()
    assert not v2_paths[name].absolute() == v1_paths[name].absolute()

    assert 'ophys_file_5.nwb' in v2_paths


@mock_s3
def test_when_data_updated(tmpdir, s3_cloud_cache_data, data_update):
    """
    Test that when a cache is instantiated after an update has
    been loaded to the dataset, the correct warning is emitted
    """
    cache_dir = pathlib.Path(tmpdir) / "test_update"
    bucket_name = "vis-behav-test-bucket"
    project_name = "vis-behav-test-proj"
    create_bucket(bucket_name,
                  project_name,
                  s3_cloud_cache_data['data'],
                  s3_cloud_cache_data['metadata'])
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                          bucket_name,
                                                          project_name)

    del cache

    client = boto3.client('s3', region_name='us-east-1')

    load_dataset(data_update['data'],
                 data_update['metadata'],
                 '0.3.0',
                 bucket_name,
                 project_name,
                 client)

    name3 = f'{project_name}_manifest_v0.3.0'
    name2 = f'{project_name}_manifest_v0.2.0'
    cmd = 'VisualBehaviorOphysProjectCache.load_manifest'
    with pytest.warns(OutdatedManifestWarning, match=name3) as warnings:
        _ = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                          bucket_name,
                                                          project_name)

    checked_msg = False
    for w in warnings.list:
        if w._category_name == 'OutdatedManifestWarning':
            msg = str(w.message)
            assert name3 in msg
            assert name2 in msg
            assert cmd in msg
            checked_msg = True
    assert checked_msg


@mock_s3
def test_load_last(tmpdir, s3_cloud_cache_data, data_update):
    """
    Test that, when a cache is instantiated over an old
    cache_dir, it loads the most recently loaded manifest,
    not the most up to date manifest
    """
    cache_dir = pathlib.Path(tmpdir) / "test_update"
    bucket_name = "vis-behav-test-bucket"
    project_name = "vis-behav-test-proj"
    create_bucket(bucket_name,
                  project_name,
                  s3_cloud_cache_data['data'],
                  s3_cloud_cache_data['metadata'])
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                          bucket_name,
                                                          project_name)

    assert cache.current_manifest() == f'{project_name}_manifest_v0.2.0.json'
    cache.load_manifest(f'{project_name}_manifest_v0.1.0.json')
    assert cache.current_manifest() == f'{project_name}_manifest_v0.1.0.json'
    del cache

    msg = 'VisualBehaviorOphysProjectCache.compare_manifests'
    with pytest.warns(OutdatedManifestWarning, match=msg):
        cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir,
                                                              bucket_name,
                                                              project_name)

    assert cache.current_manifest() == f'{project_name}_manifest_v0.1.0.json'
