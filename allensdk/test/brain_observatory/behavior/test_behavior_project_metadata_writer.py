import json
import logging
import os
import tempfile
from ast import literal_eval

import numpy as np
import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    BehaviorProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.external \
    .behavior_project_metadata_writer import \
    BehaviorProjectMetadataWriter
from allensdk.test.brain_observatory.behavior.conftest import get_resources_dir
from allensdk.test.brain_observatory.behavior.test_behavior_project_cache \
    import TempdirBehaviorCache, mock_api, session_table, behavior_table, \
    experiments_table  # noqa F401


def convert_strings_to_lists(df, is_session=True):
    df.loc[df['driver_line'].notnull(), 'driver_line'] = \
        df['driver_line'][df['driver_line'].notnull()] \
            .apply(lambda x: literal_eval(x))

    if is_session:
        df.loc[df['ophys_experiment_id'].notnull(), 'ophys_experiment_id'] = \
            df['ophys_experiment_id'][df['ophys_experiment_id'].notnull()] \
                .apply(lambda x: literal_eval(x))
        df.loc[df['ophys_container_id'].notnull(), 'ophys_container_id'] = \
            df['ophys_container_id'][df['ophys_container_id'].notnull()] \
                .apply(lambda x: literal_eval(x))


@pytest.mark.bamboo
def test_metadata():
    release_date = '2021-03-25'
    with tempfile.TemporaryDirectory() as tmp_dir:
        bpc = BehaviorProjectCache.from_lims(
            data_release_date=release_date)
        bpmw = BehaviorProjectMetadataWriter(
            behavior_project_cache=bpc,
            out_dir=tmp_dir,
            project_name='visual-behavior-ophys',
            data_release_date=release_date)
        bpmw.write_metadata()

        expected_path = os.path.join(get_resources_dir(),
                                     'project_metadata_writer',
                                     'expected')
        # test behavior
        expected = pd.read_pickle(os.path.join(expected_path,
                                               'behavior_session_table.pkl'))
        obtained = pd.read_csv(os.path.join(tmp_dir,
                                            'behavior_session_table.csv'),
                               dtype={'mouse_id': str},
                               parse_dates=['date_of_acquisition'])
        convert_strings_to_lists(df=obtained)
        pd.testing.assert_frame_equal(expected,
                                      obtained)

        # test ophys session
        expected = pd.read_pickle(os.path.join(expected_path,
                                               'ophys_session_table.pkl'))
        obtained = pd.read_csv(os.path.join(tmp_dir,
                                            'ophys_session_table.csv'),
                               dtype={'mouse_id': str},
                               parse_dates=['date_of_acquisition'])
        convert_strings_to_lists(df=obtained)
        pd.testing.assert_frame_equal(expected,
                                      obtained)

        # test ophys experiment
        expected = pd.read_pickle(os.path.join(expected_path,
                                               'ophys_experiment_table.pkl'))
        obtained = pd.read_csv(os.path.join(tmp_dir,
                                            'ophys_experiment_table.csv'),
                               dtype={'mouse_id': str},
                               parse_dates=['date_of_acquisition'])
        convert_strings_to_lists(df=obtained, is_session=False)
        pd.testing.assert_frame_equal(expected,
                                      obtained)
