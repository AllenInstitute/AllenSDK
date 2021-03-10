import os
import numpy as np
import pytest
import pandas as pd
import tempfile
import logging
from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_project_cache import (
    BehaviorProjectCache)


@pytest.fixture
def session_table():
    return (pd.DataFrame({"ophys_session_id": [1, 2, 3],
                          "ophys_experiment_id": [[4], [5, 6], [7]],
                          "date_of_acquisition": np.datetime64('2020-02-20'),
                          "reporter_line": ["aa", "bb", "cc"],
                          "driver_line": [["aa"], ["aa", "bb"], ["cc"]],
                          'full_genotype': [
                              'foo-SlcCre',
                              'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt',
                              'bar'],
                          'cre_line': [None, 'Vip-IRES-Cre', None],
                          'session_type': ['OPHYS_1_session_A',
                                           'OPHYS_1_session_A',
                                           'OPHYS_1_session_B'],
                          'specimen_id': [1, 1, 1],
                          'prior_exposures_to_session_type': [0, 1, 0],
                          'session_number': [1, 1, 1]
                          })
            .set_index("ophys_session_id"))


@pytest.fixture
def behavior_table():
    return (pd.DataFrame({"behavior_session_id": [1, 2, 3],
                          "date_of_acquisition": [
                              np.datetime64('2020-02-20'),
                              np.datetime64('2020-02-21'),
                              np.datetime64('2020-02-22')
                          ],
                          "reporter_line": ["aa", "bb", "cc"],
                          "driver_line": [["aa"], ["aa", "bb"], ["cc"]],
                          'full_genotype': [
                              'foo-SlcCre',
                              'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt',
                              'bar'],
                          'cre_line': [None, 'Vip-IRES-Cre', None],
                          'session_type': ['TRAINING_1_gratings',
                                           'TRAINING_1_gratings',
                                           'TRAINING_1_gratings'],
                          'specimen_id': [1, 1, 1],
                          'prior_exposures_to_session_type': [0, 1, 2]
                          })
            .set_index("behavior_session_id"))


@pytest.fixture
def experiments_table():
    return (pd.DataFrame({"ophys_session_id": [1, 2, 3],
                          "behavior_session_id": [1, 2, 3],
                          "ophys_experiment_id": [[4], [5, 6], [7]],
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
                          'session_type': ['OPHYS_1_session_A',
                                           'OPHYS_1_session_A',
                                           'OPHYS_1_session_B'],
                          'container_id': [1, 1, 3],
                          'specimen_id': [1, 1, 1],
                          'prior_exposures_to_session_type': [0, 1, 0],
                          'session_number': [1, 1, 1],
                          'imaging_depth': [75, 75, 75],
                          'targeted_structure': ['VISp', 'VISp', 'VISp'],
                          'indicator': ['GCaMP6f', 'GCaMP6f', 'GCaMP6f']
                          })
            .set_index("ophys_session_id"))


@pytest.fixture
def mock_api(session_table, behavior_table, experiments_table):
    class MockApi:
        def get_session_table(self):
            return session_table

        def get_behavior_only_session_table(self):
            return behavior_table

        def get_experiment_table(self):
            return experiments_table

        def get_session_data(self, ophys_session_id):
            return ophys_session_id

        def get_behavior_only_session_data(self, behavior_session_id):
            return behavior_session_id
    return MockApi


@pytest.fixture
def TempdirBehaviorCache(mock_api, request):
    temp_dir = tempfile.TemporaryDirectory()
    manifest = os.path.join(temp_dir.name, "manifest.json")
    yield BehaviorProjectCache(fetch_api=mock_api(),
                               cache=request.param,
                               manifest=manifest)
    temp_dir.cleanup()


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_session_table(TempdirBehaviorCache, session_table):
    cache = TempdirBehaviorCache
    actual = cache.get_session_table()
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_sessions").get("spec")
        assert os.path.exists(path)
    pd.testing.assert_frame_equal(session_table, actual)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_behavior_table(TempdirBehaviorCache, behavior_table):
    cache = TempdirBehaviorCache
    actual = cache.get_behavior_session_table()
    if cache.cache:
        path = cache.manifest.path_info.get("behavior_sessions").get("spec")
        assert os.path.exists(path)
    pd.testing.assert_frame_equal(behavior_table, actual)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_experiments_table(TempdirBehaviorCache, experiments_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_experiment_table()
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_experiments").get("spec")
        assert os.path.exists(path)
    pd.testing.assert_frame_equal(experiments_table, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_session_table_reads_from_cache(TempdirBehaviorCache, session_table,
                                        caplog):
    caplog.set_level(logging.INFO, logger="call_caching")
    cache = TempdirBehaviorCache
    cache.get_session_table()
    expected_first = [
        ("call_caching", logging.INFO, "Reading data from cache"),
        ("call_caching", logging.INFO, "No cache file found."),
        ("call_caching", logging.INFO, "Fetching data from remote"),
        ("call_caching", logging.INFO, "Writing data to cache"),
        ("call_caching", logging.INFO, "Reading data from cache")]
    assert expected_first == caplog.record_tuples
    caplog.clear()
    cache.get_session_table()
    assert [expected_first[0]] == caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_behavior_table_reads_from_cache(TempdirBehaviorCache, behavior_table,
                                         caplog):
    caplog.set_level(logging.INFO, logger="call_caching")
    cache = TempdirBehaviorCache
    cache.get_behavior_session_table()
    expected_first = [
        ("call_caching", logging.INFO, "Reading data from cache"),
        ("call_caching", logging.INFO, "No cache file found."),
        ("call_caching", logging.INFO, "Fetching data from remote"),
        ("call_caching", logging.INFO, "Writing data to cache"),
        ("call_caching", logging.INFO, "Reading data from cache")]
    assert expected_first == caplog.record_tuples
    caplog.clear()
    cache.get_behavior_session_table()
    assert [expected_first[0]] == caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_session_table_by_experiment(TempdirBehaviorCache):
    expected = (pd.DataFrame({"ophys_session_id": [1, 2, 2, 3],
                              "ophys_experiment_id": [4, 5, 6, 7]})
                .set_index("ophys_experiment_id"))
    actual = TempdirBehaviorCache.get_session_table(by="ophys_experiment_id")[
        ["ophys_session_id"]]
    pd.testing.assert_frame_equal(expected, actual)
