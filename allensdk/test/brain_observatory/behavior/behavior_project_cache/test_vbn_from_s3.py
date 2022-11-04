from unittest.mock import create_autospec

import pytest

from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession
from allensdk.brain_observatory.behavior.behavior_session import \
    BehaviorSession
from .utils import create_bucket, load_dataset
import boto3
from moto import mock_s3
import pathlib
import json
import semver

from allensdk.api.cloud_cache.cloud_cache import MissingLocalManifestWarning
from allensdk.api.cloud_cache.cloud_cache import OutdatedManifestWarning
from allensdk.brain_observatory.\
    behavior.behavior_project_cache.behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache


@mock_s3
def test_vbn_metadata_tables(tmpdir, vbn_s3_cloud_cache_data):
    """
    Test that we can load all metadata tables from the VBN cache

    Mostly a smoke test
    """
    data, versions = vbn_s3_cloud_cache_data
    cache_dir = pathlib.Path(tmpdir) / "test_metadata"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME
    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    cache.get_probe_table()
    cache.get_channel_table()
    cache.get_unit_table()
    cache.get_behavior_session_table()

    # make sure we are filtering ecephys sessions according
    # to abnormality correctly
    ecephys = cache.get_ecephys_session_table()
    assert len(ecephys) == 1
    assert ecephys.index[0] == 222

    abnormal = cache.get_ecephys_session_table(
                    filter_abnormalities=False)
    assert len(abnormal) == 3


@mock_s3
def test_probe_nwb_file(monkeypatch, tmpdir, vbn_s3_cloud_cache_data):
    """Tests that the probe nwb file is downloaded"""
    data, versions = vbn_s3_cloud_cache_data
    cache_dir = pathlib.Path(tmpdir) / "test_metadata"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME
    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    probe_meta_table = cache.get_probe_table()
    for probe_meta in probe_meta_table.itertuples():
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorEcephysSession, 'from_nwb_path',
                        lambda path, probe_data_path_map: probe_data_path_map)
            probe_data_path_map = \
                cache.get_ecephys_session(
                    ecephys_session_id=probe_meta.ecephys_session_id)

        probe_id = probe_meta.Index
        probe_nwb = probe_data_path_map[probe_meta.name]()
        expected_path = (cache_dir / f'{project_name}-0.{len(data)}.0' /
                         'data' / f'probe_{probe_id}_lfp.nwb')
        assert probe_nwb == expected_path
        assert expected_path.is_file()


@mock_s3
def test_manifest_methods(tmpdir, vbn_s3_cloud_cache_data):

    data, versions = vbn_s3_cloud_cache_data

    cache_dir = pathlib.Path(tmpdir) / "test_manifest_list"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME
    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    v_names = [f'{project_name}_manifest_v{i}.json' for i in versions]
    m_list = cache.list_manifest_file_names()

    assert len(m_list) == 2
    assert all([i in m_list for i in v_names])
    cache.load_manifest(v_names[0])

    # because the BehaviorProjectCloudApi automatically
    # loads the latest manifest, so the latest manifest
    # will always be the latest_downloaded_manifest
    assert cache.latest_downloaded_manifest_file() == v_names[-1]
    assert cache.latest_manifest_file() == v_names[-1]

    change_msg = cache.compare_manifests(v_names[0], v_names[-1])

    for mname in ('behavior_sessions',
                  'ecephys_sessions',
                  'probes'):
        print(change_msg)
        assert f'project_metadata/{mname} changed' in change_msg

    assert 'ecephys_file_1.nwb changed' in change_msg
    assert 'ecephys_file_3.nwb created' in change_msg


@mock_s3
def test_local_cache_construction(
    tmpdir,
    vbn_s3_cloud_cache_data,
    monkeypatch
):

    data, versions = vbn_s3_cloud_cache_data
    cache_dir = pathlib.Path(tmpdir) / "test_construction"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME
    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    v_names = [f'{project_name}_manifest_v{i}.json' for i in versions]
    cache.load_manifest(v_names[0])

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorEcephysSession, 'from_nwb_path',
                    lambda path, probe_data_path_map: create_autospec(
                        BehaviorEcephysSession, instance=True))
        cache.get_ecephys_session(ecephys_session_id=5111)
    assert cache.fetch_api.cache._downloaded_data_path.is_file()
    cache.fetch_api.cache._downloaded_data_path.unlink()
    assert not cache.fetch_api.cache._downloaded_data_path.is_file()
    del cache

    with pytest.warns(MissingLocalManifestWarning) as warnings:
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    cmd = 'VisualBehaviorNeuropixelsProjectCache.construct_local_manifest()'
    assert cmd in f'{warnings[0].message}'

    # Because, at the point where the cache was reconstitute,
    # the metadata files already existed at their expected local paths,
    # the file at _downloaded_data_path file will not have been created
    manifest_path = cache.fetch_api.cache._downloaded_data_path
    assert not manifest_path.exists()

    cache.construct_local_manifest()
    assert cache.fetch_api.cache._downloaded_data_path.is_file()

    with open(manifest_path, 'rb') as in_file:
        local_manifest = json.load(in_file)
    fnames = set([pathlib.Path(k).name for k in local_manifest])
    assert 'ecephys_file_1.nwb' in fnames
    assert len(local_manifest) == 9  # 8 metadata files and 1 data file


@mock_s3
def test_load_out_of_date_manifest(
    tmpdir,
    vbn_s3_cloud_cache_data,
    monkeypatch
):
    """
    Test that VisualBehaviorNeuropixelsProjectCache can load a
    manifest other than the latest and download files
    from that manifest.
    """
    data, versions = vbn_s3_cloud_cache_data

    cache_dir = pathlib.Path(tmpdir) / "test_linkage"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME
    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    v_names = [f'{project_name}_manifest_v{i}.json' for i in versions]
    cache.load_manifest(v_names[0])
    for sess_id in (333, 444):
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorSession, 'from_nwb_path',
                        lambda path: create_autospec(
                            BehaviorSession, instance=True))
            cache.get_behavior_session(behavior_session_id=sess_id)
    for ses_id in (5111, 5112):
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorEcephysSession, 'from_nwb_path',
                        lambda path, probe_data_path_map: create_autospec(
                            BehaviorEcephysSession, instance=True))
            cache.get_ecephys_session(ecephys_session_id=ses_id)

    v1_dir = cache_dir / f'{project_name}-{versions[0]}/data'

    # Check that all expected file were downloaded
    dir_glob = v1_dir.glob('*')
    file_names = set()
    file_contents = {}
    for p in dir_glob:
        file_names.add(p.name)
        with open(p, 'rb') as in_file:
            data = in_file.read()
        file_contents[p.name] = data
    expected = {'ecephys_file_1.nwb', 'ecephys_file_2.nwb'}

    assert file_names == expected

    expected = {}
    expected['ecephys_file_1.nwb'] = b'abcde'
    expected['ecephys_file_2.nwb'] = b'fghijk'

    assert file_contents == expected


@mock_s3
@pytest.mark.parametrize("delete_cache", [True, False])
def test_file_linkage(
    tmpdir,
    vbn_s3_cloud_cache_data,
    delete_cache,
    monkeypatch
):
    """
    Test that symlinks are used where appropriate

    if delete_cache == True, will delete the local cache
    file between loading v1 and v2 manifests, then run
    construct_local_cache() to make sure that the symlinks
    are still properly constructed
    """
    data, versions = vbn_s3_cloud_cache_data
    cache_dir = pathlib.Path(tmpdir) / "test_linkage"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME

    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    v_names = [f'{project_name}_manifest_v{i}.json' for i in versions]
    v_dirs = [cache_dir / f'{project_name}-{i}/data' for i in versions]

    assert cache.current_manifest() == v_names[-1]
    assert cache.list_all_downloaded_manifests() == [v_names[-1]]

    cache.load_manifest(v_names[0])
    assert cache.current_manifest() == v_names[0]
    assert cache.list_all_downloaded_manifests() == v_names

    for sess_id in (333, 444):
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorSession, 'from_nwb_path',
                        lambda path: create_autospec(
                            BehaviorSession, instance=True))
            cache.get_behavior_session(behavior_session_id=sess_id)
    for sess_id in (5111, 5112):
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorEcephysSession, 'from_nwb_path',
                        lambda path, probe_data_path_map: create_autospec(
                            BehaviorEcephysSession, instance=True))
            cache.get_ecephys_session(ecephys_session_id=sess_id)

    v1_glob = v_dirs[0].glob('*')
    v1_paths = {}
    for p in v1_glob:
        v1_paths[p.name] = p

    if delete_cache:
        local_cache = cache.fetch_api.cache._downloaded_data_path
        assert local_cache.is_file()
        local_cache.unlink()
        assert not local_cache.is_file()
        del cache

        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)
        cache.construct_local_manifest()

    cache.load_manifest(v_names[-1])
    assert cache.current_manifest() == v_names[-1]
    for sess_id in (777, 888):
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorSession, 'from_nwb_path',
                        lambda path: create_autospec(
                            BehaviorSession, instance=True))
            cache.get_behavior_session(behavior_session_id=sess_id)
    for sess_id in (222, 333):
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorEcephysSession, 'from_nwb_path',
                        lambda path, probe_data_path_map: create_autospec(
                            BehaviorEcephysSession, instance=True))
            cache.get_ecephys_session(ecephys_session_id=sess_id)

    v2_glob = v_dirs[-1].glob('*')
    v2_paths = {}
    for p in v2_glob:
        v2_paths[p.name] = p

    # check symlinks
    for name in ('ecephys_file_2.nwb',):

        assert v2_paths[name].is_symlink()
        assert v2_paths[name].resolve() == v1_paths[name].resolve()
        assert v2_paths[name].absolute() != v1_paths[name].absolute()

    name = 'ecephys_file_1.nwb'
    assert not v2_paths[name].is_symlink()
    assert not v2_paths[name].absolute() == v1_paths[name].absolute()


@mock_s3
def test_when_data_updated(tmpdir, vbn_s3_cloud_cache_data, data_update):
    """
    Test that when a cache is instantiated after an update has
    been loaded to the dataset, the correct warning is emitted
    """
    data, versions = vbn_s3_cloud_cache_data
    cache_dir = pathlib.Path(tmpdir) / "test_update"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME
    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    del cache

    client = boto3.client('s3', region_name='us-east-1')

    later_version = str(semver.parse_version_info(versions[-1]).bump_minor())
    load_dataset(data_update['data'],
                 data_update['metadata'],
                 later_version,
                 bucket_name,
                 project_name,
                 client)

    name3 = f'{project_name}_manifest_v{later_version}'
    name2 = f'{project_name}_manifest_v{versions[-1]}'

    cmd = 'VisualBehaviorNeuropixelsProjectCache.load_manifest'
    with pytest.warns(OutdatedManifestWarning, match=name3) as warnings:
        VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

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
def test_load_last(tmpdir, vbn_s3_cloud_cache_data, data_update):
    """
    Test that, when a cache is instantiated over an old
    cache_dir, it loads the most recently loaded manifest,
    not the most up to date manifest
    """
    data, versions = vbn_s3_cloud_cache_data
    cache_dir = pathlib.Path(tmpdir) / "test_update"
    bucket_name = VisualBehaviorNeuropixelsProjectCache.BUCKET_NAME
    project_name = VisualBehaviorNeuropixelsProjectCache.PROJECT_NAME
    create_bucket(bucket_name,
                  project_name,
                  data['data'],
                  data['metadata'])
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    v_names = [f'{project_name}_manifest_v{i}.json' for i in versions]

    assert cache.current_manifest() == v_names[-1]
    cache.load_manifest(v_names[0])
    assert cache.current_manifest() == v_names[0]
    del cache

    msg = 'VisualBehaviorNeuropixelsProjectCache.compare_manifests'
    with pytest.warns(OutdatedManifestWarning, match=msg):
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir)

    assert cache.current_manifest() == v_names[0]
