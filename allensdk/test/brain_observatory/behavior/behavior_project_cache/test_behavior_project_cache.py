import os
import numpy as np
import pytest
import pandas as pd
import tempfile
import logging

from allensdk.brain_observatory.behavior.behavior_project_cache \
    import VisualBehaviorOphysProjectCache
from allensdk.test.brain_observatory.behavior.conftest import get_resources_dir


@pytest.fixture
def session_table():
    return (pd.DataFrame({"behavior_session_id": [3],
                          "ophys_experiment_id": [[5, 6]],
                          "date_of_acquisition": np.datetime64('2020-02-20'),
                          'session_type': ['OPHYS_1_images_A'],
                          }, index=pd.Index([1], name='ophys_session_id'))
            )


@pytest.fixture
def behavior_table():
    return (pd.DataFrame({"behavior_session_id": [1, 2, 3],
                          "foraging_id": [1, 2, 3],
                          "date_of_acquisition": [
                              np.datetime64('2020-02-20'),
                              np.datetime64('2020-02-21'),
                              np.datetime64('2020-02-22')
                          ],
                          "reporter_line": ["Ai93(TITL-GCaMP6f)",
                                            "Ai93(TITL-GCaMP6f)",
                                            "Ai93(TITL-GCaMP6f)"],
                          "driver_line": [["aa"], ["aa", "bb"], ["cc"]],
                          'full_genotype': [
                              'foo-SlcCre',
                              'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt',
                              'bar'],
                          'cre_line': [None, 'Vip-IRES-Cre', None],
                          'session_type': ['TRAINING_1_gratings',
                                           'TRAINING_1_gratings',
                                           'OPHYS_1_images_A'],
                          'mouse_id': [1, 1, 1]
                          })
            .set_index("behavior_session_id"))


@pytest.fixture
def experiments_table():
    return (pd.DataFrame({"ophys_session_id": [1, 2, 3],
                          "behavior_session_id": [1, 2, 3],
                          "ophys_experiment_id": [1, 2, 3],
                          "date_of_acquisition": [
                              np.datetime64('2020-02-20'),
                              np.datetime64('2020-02-21'),
                              np.datetime64('2020-02-22')
                          ],
                          'session_type': ['TRAINING_1_gratings',
                                           'TRAINING_1_gratings',
                                           'OPHYS_1_images_A'],
                          'imaging_depth': [75, 75, 75],
                          'targeted_structure': ['VISp', 'VISp', 'VISp']
                          })
            .set_index("ophys_experiment_id"))


@pytest.fixture
def mock_api(session_table, behavior_table, experiments_table):
    class MockApi:
        def get_ophys_session_table(self):
            return session_table

        def get_behavior_session_table(self):
            return behavior_table

        def get_ophys_experiment_table(self):
            return experiments_table

        def get_session_data(self, ophys_session_id):
            return ophys_session_id

        def get_behavior_stage_parameters(self, foraging_ids):
            return {x: {} for x in foraging_ids}

    return MockApi


@pytest.fixture
def TempdirBehaviorCache(mock_api, request):
    temp_dir = tempfile.TemporaryDirectory()
    manifest = os.path.join(temp_dir.name, "manifest.json")
    yield VisualBehaviorOphysProjectCache(fetch_api=mock_api(),
                                          cache=request.param,
                                          manifest=manifest)
    temp_dir.cleanup()


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_ophys_session_table(TempdirBehaviorCache, session_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_ophys_session_table()
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_sessions").get("spec")
        assert os.path.exists(path)

    expected_path = os.path.join(get_resources_dir(), 'project_metadata',
                                 'expected')
    expected = pd.read_pickle(os.path.join(expected_path,
                                           'ophys_session_table.pkl'))

    pd.testing.assert_frame_equal(expected, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_behavior_table(TempdirBehaviorCache, behavior_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_behavior_session_table()
    if cache.cache:
        path = cache.manifest.path_info.get("behavior_sessions").get("spec")
        assert os.path.exists(path)

    expected_path = os.path.join(get_resources_dir(), 'project_metadata',
                                 'expected')
    expected = pd.read_pickle(os.path.join(expected_path,
                                           'behavior_session_table.pkl'))

    pd.testing.assert_frame_equal(expected, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_experiments_table(TempdirBehaviorCache, experiments_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_ophys_experiment_table()
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_experiments").get("spec")
        assert os.path.exists(path)

    expected_path = os.path.join(get_resources_dir(), 'project_metadata',
                                 'expected')
    expected = pd.read_pickle(os.path.join(expected_path,
                                           'ophys_experiment_table.pkl'))

    pd.testing.assert_frame_equal(expected, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_session_table_reads_from_cache(TempdirBehaviorCache, session_table,
                                        caplog):
    caplog.set_level(logging.INFO, logger="call_caching")
    cache = TempdirBehaviorCache
    cache.get_ophys_session_table()
    expected_first = [
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'No cache file found.'),
        ('call_caching', logging.INFO, 'Fetching data from remote'),
        ('call_caching', logging.INFO, 'Writing data to cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'No cache file found.'),
        ('call_caching', logging.INFO, 'Fetching data from remote'),
        ('call_caching', logging.INFO, 'Writing data to cache'),
        ('call_caching', logging.INFO, 'Reading data from cache')]
    assert expected_first == caplog.record_tuples
    caplog.clear()
    cache.get_ophys_session_table()
    assert [expected_first[0], expected_first[-1]] == caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_behavior_table_reads_from_cache(TempdirBehaviorCache, behavior_table,
                                         caplog):
    caplog.set_level(logging.INFO, logger="call_caching")
    cache = TempdirBehaviorCache
    cache.get_behavior_session_table()
    expected_first = [
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'No cache file found.'),
        ('call_caching', logging.INFO, 'Fetching data from remote'),
        ('call_caching', logging.INFO, 'Writing data to cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'No cache file found.'),
        ('call_caching', logging.INFO, 'Fetching data from remote'),
        ('call_caching', logging.INFO, 'Writing data to cache'),
        ('call_caching', logging.INFO, 'Reading data from cache')]
    assert expected_first == caplog.record_tuples
    caplog.clear()
    cache.get_behavior_session_table()
    assert [expected_first[0], expected_first[-1]] == caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_ophys_session_table_by_experiment(TempdirBehaviorCache):
    expected = (pd.DataFrame({"ophys_session_id": [1, 1],
                              "ophys_experiment_id": [5, 6]})
                .set_index("ophys_experiment_id"))
    actual = TempdirBehaviorCache.get_ophys_session_table(
        index_column="ophys_experiment_id")[
        ["ophys_session_id"]]
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_cloud_manifest_errors(TempdirBehaviorCache):
    """
    Test that methods which should not exist for BehaviorProjectCaches
    that are not backed by CloudCaches raise NotImplementedError
    """
    msg = 'Method {mname} does not exist for this '
    msg += 'VisualBehaviorOphysProjectCache, which is based on MockApi'
    with pytest.raises(NotImplementedError,
                       match=msg.format(mname='construct_local_manifest')):
        TempdirBehaviorCache.construct_local_manifest()

    with pytest.raises(NotImplementedError,
                       match=msg.format(mname='compare_manifests')):
        TempdirBehaviorCache.compare_manifests('a', 'b')

    with pytest.raises(NotImplementedError,
                       match=msg.format(mname='load_latest_manifest')):
        TempdirBehaviorCache.load_latest_manifest()

    this_msg = msg.format(mname='latest_downloaded_manifest_file')
    with pytest.raises(NotImplementedError,
                       match=this_msg):
        TempdirBehaviorCache.latest_downloaded_manifest_file()

    with pytest.raises(NotImplementedError,
                       match=msg.format(mname='latest_manifest_file')):
        TempdirBehaviorCache.latest_manifest_file()

    with pytest.raises(NotImplementedError,
                       match=msg.format(mname='load_manifest')):
        TempdirBehaviorCache.load_manifest('a')

    with pytest.raises(NotImplementedError,
                       match=msg.format(mname='current_manifest')):
        TempdirBehaviorCache.current_manifest()

    this_msg = msg.format(mname='list_all_downloaded_manifests')
    with pytest.raises(NotImplementedError,
                       match=this_msg):
        TempdirBehaviorCache.list_all_downloaded_manifests()

    this_msg = msg.format(mname='list_manifest_file_names')
    with pytest.raises(NotImplementedError,
                       match=this_msg):
        TempdirBehaviorCache.list_manifest_file_names()
