import os
import numpy as np
import pytest
import pandas as pd
import tempfile
import logging

from allensdk.brain_observatory.behavior.behavior_project_cache \
    import BehaviorProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.external\
    .behavior_project_metadata import \
    BehaviorProjectMetadataWriter


@pytest.fixture
def session_table():
    return (pd.DataFrame({"behavior_session_id": [3],
                          "foraging_id": [1],
                          "ophys_experiment_id": [[5, 6]],
                          "date_of_acquisition": np.datetime64('2020-02-20'),
                          "reporter_line": ["Ai93(TITL-GCaMP6f)"],
                          "driver_line": [["aa"]],
                          'full_genotype': [
                              'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt',
                              ],
                          'cre_line': ['Vip-IRES-Cre'],
                          'session_type': ['OPHYS_1_images_A'],
                          'mouse_id': [1],
                          'session_number': [1],
                          'indicator': ['GCaMP6f']
                          }, index=pd.Index([1], name='ophys_session_id'))
            )


@pytest.fixture
def behavior_table():
    return (pd.DataFrame({"behavior_session_id": [1, 2, 3],
                          "ophys_session_id": [2, 1, 3],
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
                          'session_number': [None, None, 1],
                          'mouse_id': [1, 1, 1],
                          'prior_exposures_to_session_type': [0, 1, 0],
                          'prior_exposures_to_image_set': [
                              np.nan, np.nan, 0],
                          'prior_exposures_to_omissions': [0, 0, 0],
                          'indicator': ['GCaMP6f', 'GCaMP6f', 'GCaMP6f']
                          })
            .set_index("behavior_session_id"))


@pytest.fixture
def experiments_table():
    return (pd.DataFrame({"ophys_session_id": [1, 2, 3],
                          "behavior_session_id": [1, 2, 3],
                          "foraging_id": [1, 2, 3],
                          "ophys_experiment_id": [1, 2, 3],
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
                          'mouse_id': [1, 1, 1],
                          'session_number': [None, None, 1],
                          'imaging_depth': [75, 75, 75],
                          'targeted_structure': ['VISp', 'VISp', 'VISp'],
                          'indicator': ['GCaMP6f', 'GCaMP6f', 'GCaMP6f']
                          })
            .set_index("ophys_experiment_id"))


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

        def get_behavior_stage_parameters(self, foraging_ids):
            return {x: {} for x in foraging_ids}
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
    obtained = cache.get_session_table()
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_sessions").get("spec")
        assert os.path.exists(path)

    # These get merged in
    session_table['prior_exposures_to_session_type'] = [0]
    session_table['prior_exposures_to_image_set'] = [0.0]
    session_table['prior_exposures_to_omissions'] = [0]

    pd.testing.assert_frame_equal(session_table, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_behavior_table(TempdirBehaviorCache, behavior_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_behavior_session_table()
    if cache.cache:
        path = cache.manifest.path_info.get("behavior_sessions").get("spec")
        assert os.path.exists(path)
    pd.testing.assert_frame_equal(behavior_table, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True, False], indirect=True)
def test_get_experiments_table(TempdirBehaviorCache, experiments_table):
    cache = TempdirBehaviorCache
    obtained = cache.get_experiment_table()
    if cache.cache:
        path = cache.manifest.path_info.get("ophys_experiments").get("spec")
        assert os.path.exists(path)

    # These get merged in
    experiments_table['prior_exposures_to_session_type'] = [0, 1, 0]
    experiments_table['prior_exposures_to_image_set'] = [np.nan, np.nan, 0]
    experiments_table['prior_exposures_to_omissions'] = [0, 0, 0]

    pd.testing.assert_frame_equal(experiments_table, obtained)


@pytest.mark.parametrize("TempdirBehaviorCache", [True], indirect=True)
def test_session_table_reads_from_cache(TempdirBehaviorCache, session_table,
                                        caplog):
    caplog.set_level(logging.INFO, logger="call_caching")
    cache = TempdirBehaviorCache
    cache.get_session_table()
    expected_first = [
        ('call_caching', 20, 'Reading data from cache'),
        ('call_caching', 20, 'No cache file found.'),
        ('call_caching', 20, 'Fetching data from remote'),
        ('call_caching', 20, 'Writing data to cache'),
        ('call_caching', 20, 'Reading data from cache'),
        ('call_caching', 20, 'Reading data from cache'),
        ('call_caching', 20, 'No cache file found.'),
        ('call_caching', 20, 'Fetching data from remote'),
        ('call_caching', 20, 'Writing data to cache'),
        ('call_caching', 20, 'Reading data from cache')]
    assert expected_first == caplog.record_tuples
    caplog.clear()
    cache.get_session_table()
    assert [expected_first[0], expected_first[-1]] == caplog.record_tuples


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
    expected = (pd.DataFrame({"ophys_session_id": [1, 1],
                              "ophys_experiment_id": [5, 6]})
                .set_index("ophys_experiment_id"))
    actual = TempdirBehaviorCache.get_session_table(
        index_column="ophys_experiment_id")[
        ["ophys_session_id"]]
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("TempdirBehaviorCache", [False], indirect=True)
@pytest.mark.parametrize("which",
                         ('behavior_session_table', 'ophys_session_table',
                          'ophys_experiment_table'))
def test_write_behavior_sessions(TempdirBehaviorCache, monkeypatch, which):
    cache = TempdirBehaviorCache

    def _get_release_files(self, file_type):
        if file_type == 'BehaviorNwb':
            return pd.DataFrame({
                'file_id': [1],
                'isilon_filepath': ['/tmp/behavior_session.nwb']
            }, index=pd.Index([1], name='behavior_session_id'))
        else:
            return pd.DataFrame({
                'file_id': [2],
                'isilon_filepath': ['/tmp/imaging_plane.nwb']
            }, index=pd.Index([1], name='ophys_experiment_id'))

    def _get_ophys_sessions_from_ophys_experiments(self,
                                                   ophys_experiment_ids=None):
        return pd.Series([1])

    with tempfile.TemporaryDirectory() as temp_dir:
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorProjectMetadataWriter,
                        '_get_release_files',
                        _get_release_files)
            ctx.setattr(BehaviorProjectMetadataWriter,
                        '_get_ophys_sessions_from_ophys_experiments',
                        _get_ophys_sessions_from_ophys_experiments)
            bpmw = BehaviorProjectMetadataWriter(behavior_project_cache=cache,
                                                 out_dir=temp_dir)

            if which == 'behavior_session_table':
                bpmw._write_behavior_sessions()
                filename = 'behavior_session_table.csv'
                df = pd.read_csv(os.path.join(temp_dir, filename))

                assert df.shape[0] == 2
                assert df[df['behavior_session_id'] == 1]\
                    .iloc[0]['file_id'] == 1
                assert np.isnan(df[df['ophys_session_id'] == 1]
                    .iloc[0]['file_id'])

            elif which == 'ophys_session_table':
                bpmw._write_ophys_sessions()
                filename = 'ophys_session_table.csv'
                df = pd.read_csv(os.path.join(temp_dir, filename))

                assert df.shape[0] == 1
                assert 'file_id' not in df.columns and \
                       'isilon_filepath' not in df.columns
            elif which == 'ophys_experiment_table':
                bpmw._write_ophys_experiments()
                filename = 'ophys_experiment_table.csv'

                df = pd.read_csv(os.path.join(temp_dir, filename))

                assert df.shape[0] == 1
                assert df[df['ophys_experiment_id'] == 1]\
                    .iloc[0]['file_id'] == 2
            else:
                raise ValueError(f'{which} not understood')
