import pytest
import pandas as pd
import logging
import os


def safe_df_comparison(expected: pd.DataFrame, obtained: pd.DataFrame):
    """
    Compare two dataframes in a way that is agnostic to column order
    and datatype of NULL values
    """
    msg = ''
    if len(obtained.columns) != len(expected.columns):
        msg += 'column mis-match\n'
        msg += 'obtained columns\n'
        msg += f'{obtained.columns}\n'
        msg += 'expected columns\n'
        msg += f'{expected.columns}\n'

        missing_from_obtained = []
        for c in expected.columns:
            if c not in obtained.columns:
                missing_from_obtained.append(c)
        missing_from_expected = []
        for c in obtained.columns:
            if c not in expected.columns:
                missing_from_expected.append(c)
        msg += f'missing from obtained\n{missing_from_obtained}\n'
        msg += f'missing from expected\n{missing_from_expected}\n'
        raise RuntimeError(msg)

    if not expected.index.equals(obtained.index):
        msg += 'index mis-match\n'
        msg += 'expected index\n'
        msg += f'{expected.index}\n'
        msg += 'obtained index\n'
        msg += f'{obtained.index}\n'
        raise RuntimeError(msg)

    for col in expected.columns:
        expected_null = expected[col].isnull()
        obtained_null = obtained[col].isnull()
        if not expected_null.equals(obtained_null):
            msg += f'\n{col} not null at same point in '
            msg += 'obtained and expected\n'
            continue
        expected_valid = expected[~expected_null]
        obtained_valid = obtained[~obtained_null]
        if not expected_valid.index.equals(obtained_valid.index):
            msg += '\nindex mismatch in non-null when checking '
            msg += f'{col}\n'
        for index_val in expected_valid.index.values:
            e = expected_valid.at[index_val, col]
            o = obtained_valid.at[index_val, col]
            if not e == o:
                msg += f'\n{col}\n'
                msg += f'expected: {e}\n'
                msg += f'obtained: {o}\n'
    if msg != '':
        raise RuntimeError(msg)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_ophys_session_table(TempdirBehaviorCache,
                                 expected_ophys_session_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_ophys_session_table()
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_sessions").get("spec")
        assert os.path.exists(path)

    safe_df_comparison(expected_ophys_session_table,
                       obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_behavior_table(TempdirBehaviorCache,
                            expected_behavior_session_table,
                            container_state_lookup,
                            experiment_state_lookup,
                            ophys_experiment_to_container_map):
    cache = TempdirBehaviorCache
    obtained = cache.get_behavior_session_table()
    expected = expected_behavior_session_table
    if cache.cache:
        path = cache.manifest.path_info.get("behavior_sessions").get("spec")
        assert os.path.exists(path)

    safe_df_comparison(expected, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_experiments_table(TempdirBehaviorCache,
                               expected_experiments_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_ophys_experiment_table(passed_only=True)
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_experiments").get("spec")
        assert os.path.exists(path)

    safe_df_comparison(expected_experiments_table, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_session_table_reads_from_cache(TempdirBehaviorCache,
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
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'No cache file found.'),
        ('call_caching', logging.INFO, 'Fetching data from remote'),
        ('call_caching', logging.INFO, 'Writing data to cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'Reading data from cache')]
    assert expected_first == caplog.record_tuples
    caplog.clear()
    cache.get_ophys_session_table()
    assert [expected_first[0]]*4 == caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_behavior_table_reads_from_cache(TempdirBehaviorCache,
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
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'No cache file found.'),
        ('call_caching', logging.INFO, 'Fetching data from remote'),
        ('call_caching', logging.INFO, 'Writing data to cache'),
        ('call_caching', logging.INFO, 'Reading data from cache'),
        ('call_caching', logging.INFO, 'Reading data from cache')]
    assert expected_first == caplog.record_tuples
    caplog.clear()
    cache.get_behavior_session_table()
    assert [expected_first[0]]*4 == caplog.record_tuples


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_ophys_session_table_by_experiment(TempdirBehaviorCache,
                                               expected_ophys_session_table):

    raw = expected_ophys_session_table[['ophys_experiment_id']]
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
