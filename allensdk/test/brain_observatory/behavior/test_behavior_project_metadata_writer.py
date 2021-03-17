import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.behavior_project_cache.external\
    .behavior_project_metadata_writer import \
    BehaviorProjectMetadataWriter
from allensdk.test.brain_observatory.behavior.test_behavior_project_cache \
    import TempdirBehaviorCache, mock_api, session_table, behavior_table, \
    experiments_table  #noqa F401


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


@pytest.mark.parametrize("TempdirBehaviorCache", [False], indirect=True)
@pytest.mark.parametrize("which",
                         ('behavior_session_table', 'ophys_session_table',
                          'ophys_experiment_table'))
def test_write_metadata_tables(TempdirBehaviorCache, monkeypatch, which):
    """Tests writing all metadata tables"""
    cache = TempdirBehaviorCache

    with tempfile.TemporaryDirectory() as temp_dir:
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorProjectMetadataWriter,
                        '_get_release_files',
                        _get_release_files)
            ctx.setattr(BehaviorProjectMetadataWriter,
                        '_get_ophys_sessions_from_ophys_experiments',
                        _get_ophys_sessions_from_ophys_experiments)
            bpmw = BehaviorProjectMetadataWriter(behavior_project_cache=cache,
                                                 out_dir=temp_dir,
                                                 project_name='test')

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


@pytest.mark.parametrize("TempdirBehaviorCache", [False], indirect=True)
def test_write_manifest(TempdirBehaviorCache, monkeypatch):
    """Tests writing manifest json"""
    cache = TempdirBehaviorCache

    with tempfile.TemporaryDirectory() as temp_dir:
        with monkeypatch.context() as ctx:
            ctx.setattr(BehaviorProjectMetadataWriter,
                        '_get_release_files',
                        _get_release_files)
            ctx.setattr(BehaviorProjectMetadataWriter,
                        '_get_ophys_sessions_from_ophys_experiments',
                        _get_ophys_sessions_from_ophys_experiments)
            bpmw = BehaviorProjectMetadataWriter(behavior_project_cache=cache,
                                                 out_dir=temp_dir,
                                                 project_name='test')
            bpmw.write_metadata()

            with open(os.path.join(temp_dir, 'manifest.json')) as f:
                manifest = json.loads(f.read())

            assert bpmw._get_release_files(file_type='BehaviorNwb')\
                ['isilon_filepath'].isin(manifest['data_files']).all()
            assert bpmw._get_release_files(file_type='BehaviorOphysNwb')\
                ['isilon_filepath'].isin(manifest['data_files']).all()
            assert [x for x in os.listdir(temp_dir) if x.endswith('.csv')] == \
                   [os.path.basename(x) for x in manifest['metadata_files']]
