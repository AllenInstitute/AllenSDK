import os
import pytest
import pandas as pd
import tempfile
import logging
import time
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    BehaviorProjectCache)
from allensdk.core.exceptions import MissingDataError


@pytest.fixture
def session_table():
    return (pd.DataFrame({"ophys_session_id": [1, 2, 3],
                          "ophys_experiment_id": [[4], [5, 6], [7]],
                          "reporter_line": [["aa"], ["aa", "bb"], ["cc"]],
                          "driver_line": [["aa"], ["aa", "bb"], ["cc"]]})
            .set_index("ophys_session_id"))


@pytest.fixture
def behavior_table():
    return (pd.DataFrame({"behavior_session_id": [1, 2, 3],
                          "reporter_line": [["aa"], ["aa", "bb"], ["cc"]],
                          "driver_line": [["aa"], ["aa", "bb"], ["cc"]]})
            .set_index("behavior_session_id"))


@pytest.fixture
def mock_api(session_table, behavior_table):
    class MockApi:
        def get_session_table(self):
            return session_table

        def get_behavior_only_session_table(self):
            return behavior_table

        def get_session_data(self, ophys_session_id):
            return ophys_session_id

        def get_behavior_only_session_data(self, behavior_session_id):
            return behavior_session_id
    return MockApi


@pytest.fixture
def TempdirBehaviorCache(mock_api):
    temp_dir = tempfile.TemporaryDirectory()
    manifest = os.path.join(temp_dir.name, "manifest.json")
    yield BehaviorProjectCache(fetch_api=mock_api(),
                               manifest=manifest)
    temp_dir.cleanup()


def test_get_session_table(TempdirBehaviorCache, session_table):
    cache = TempdirBehaviorCache
    actual = cache.get_session_table()
    path = cache.manifest.path_info.get("ophys_sessions").get("spec")
    assert os.path.exists(path)
    pd.testing.assert_frame_equal(session_table, actual)


def test_get_behavior_table(TempdirBehaviorCache, behavior_table):
    cache = TempdirBehaviorCache
    actual = cache.get_behavior_session_table()
    path = cache.manifest.path_info.get("behavior_sessions").get("spec")
    assert os.path.exists(path)
    pd.testing.assert_frame_equal(behavior_table, actual)


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


def test_behavior_session_fails_fixed_if_no_cache(TempdirBehaviorCache):
    cache = TempdirBehaviorCache
    with pytest.raises(MissingDataError):
        cache.get_behavior_session_data(1, fixed=True)
    cache.get_behavior_session_data(1)
    # Also fails if there is a cache, but the id is not contained therein
    with pytest.raises(MissingDataError):
        cache.get_behavior_session_data(2, fixed=True)


def test_session_fails_fixed_if_no_cache(TempdirBehaviorCache):
    cache = TempdirBehaviorCache
    with pytest.raises(MissingDataError):
        cache.get_session_data(1, fixed=True)
    cache.get_session_data(1)
    # Also fails if there is a cache, but the id is not contained therein
    with pytest.raises(MissingDataError):
        cache.get_session_data(2, fixed=True)


def test_get_session_table_by_experiment(TempdirBehaviorCache):
    expected = (pd.DataFrame({"ophys_session_id": [1, 2, 2, 3],
                              "ophys_experiment_id": [4, 5, 6, 7]})
                .set_index("ophys_experiment_id"))
    actual = TempdirBehaviorCache.get_session_table(by="ophys_experiment_id")[
        ["ophys_session_id"]]
    pd.testing.assert_frame_equal(expected, actual)


def test_write_behavior_log(TempdirBehaviorCache):
    expected_cols = ["behavior_session_id", "created_at", "updated_at"]
    expected_ids = [1, 2]
    expected_times = [False, True]
    cache = TempdirBehaviorCache
    cache.get_behavior_session_data(1)
    cache.get_behavior_session_data(2)
    time.sleep(1)
    cache.get_behavior_session_data(1)
    path = cache.manifest.path_info.get("behavior_analysis_log").get("spec")
    # Log exists
    assert os.path.exists(path)
    actual = pd.read_csv(path)
    # columns exist
    assert list(actual) == expected_cols
    # ids exist
    assert actual["behavior_session_id"].values.tolist() == expected_ids
    # first one should have updated different than created since accessed 2x
    assert ((actual["created_at"] == actual["updated_at"]).values.tolist()
            == expected_times)


def test_write_session_log(TempdirBehaviorCache):
    expected_cols = ["ophys_experiment_id", "created_at", "updated_at"]
    expected_ids = [1, 2]
    expected_times = [False, True]
    cache = TempdirBehaviorCache
    cache.get_session_data(1)
    cache.get_session_data(2)
    time.sleep(1)
    cache.get_session_data(1)
    path = cache.manifest.path_info.get("ophys_analysis_log").get("spec")
    # Log exists
    assert os.path.exists(path)
    actual = pd.read_csv(path)
    # columns exist
    assert list(actual) == expected_cols
    # ids exist
    assert actual["ophys_experiment_id"].values.tolist() == expected_ids
    # first one should have updated different than created since accessed 2x
    assert ((actual["created_at"] == actual["updated_at"]).values.tolist()
            == expected_times)
