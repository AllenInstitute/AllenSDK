import pytest
import pandas as pd
import logging


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_session_table_reads_from_cache(TempdirBehaviorCache,
                                        caplog):
    caplog.set_level(logging.INFO, logger="call_caching")
    cache = TempdirBehaviorCache
    cache.get_ophys_session_table()
    reading_tuple = ('call_caching', logging.INFO, 'Reading data from cache')
    no_file_tuple = ('call_caching', logging.INFO, 'No cache file found.')
    writing_tuple = ('call_caching', logging.INFO, 'Writing data to cache')
    assert reading_tuple in caplog.record_tuples
    assert no_file_tuple in caplog.record_tuples
    assert writing_tuple in caplog.record_tuples

    caplog.clear()
    cache.get_ophys_session_table()
    assert reading_tuple in caplog.record_tuples
    assert no_file_tuple not in caplog.record_tuples
    assert writing_tuple not in caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_behavior_table_reads_from_cache(TempdirBehaviorCache,
                                         caplog):
    caplog.set_level(logging.INFO, logger="call_caching")
    cache = TempdirBehaviorCache
    cache.get_behavior_session_table()
    reading_tuple = ('call_caching', logging.INFO, 'Reading data from cache')
    no_file_tuple = ('call_caching', logging.INFO, 'No cache file found.')
    writing_tuple = ('call_caching', logging.INFO, 'Writing data to cache')
    assert reading_tuple in caplog.record_tuples
    assert no_file_tuple in caplog.record_tuples
    assert writing_tuple in caplog.record_tuples

    caplog.clear()
    cache.get_behavior_session_table()
    assert reading_tuple in caplog.record_tuples
    assert no_file_tuple not in caplog.record_tuples
    assert writing_tuple not in caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_ophys_session_table_by_experiment(TempdirBehaviorCache,
                                               expected_ophys_session_table):

    raw = expected_ophys_session_table['df'][['ophys_experiment_id']]
    data = []
    for session_id, exp_id_list in zip(raw.index.values,
                                       raw.ophys_experiment_id.values):
        for exp_id in exp_id_list:
            data.append({'ophys_session_id': session_id,
                         'ophys_experiment_id': exp_id})

    expected = pd.DataFrame(data).set_index('ophys_experiment_id')

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
