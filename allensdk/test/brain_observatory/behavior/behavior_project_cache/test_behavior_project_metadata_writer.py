import os
import tempfile
from ast import literal_eval

import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.external \
    .behavior_project_metadata_writer import \
    BehaviorProjectMetadataWriter
from allensdk.test.brain_observatory.behavior.conftest import get_resources_dir


def convert_strings_to_lists(df, is_session=True):
    """Lists when inside dataframe and written using .to_csv
    get written as string literals. Need to parse out lists"""
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


def sort_df(df: pd.DataFrame, sort_col: str):
    """Sorts df for comparison"""
    df = df.sort_values(sort_col)\
        .reset_index()\
        .drop('index', axis=1)
    df = df[df.columns.sort_values()]
    return df


@pytest.mark.requires_bamboo
def test_metadata():
    release_date = '2021-03-25'
    with tempfile.TemporaryDirectory() as tmp_dir:
        bpc = VisualBehaviorOphysProjectCache.from_lims(
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
        expected = sort_df(df=expected, sort_col='behavior_session_id')
        obtained = pd.read_csv(os.path.join(tmp_dir,
                                            'behavior_session_table.csv'),
                               dtype={'mouse_id': str},
                               parse_dates=['date_of_acquisition'])
        obtained = sort_df(df=obtained, sort_col='behavior_session_id')
        convert_strings_to_lists(df=obtained)
        pd.testing.assert_frame_equal(expected,
                                      obtained)

        # test ophys session
        expected = pd.read_pickle(os.path.join(expected_path,
                                               'ophys_session_table.pkl'))
        expected = sort_df(df=expected, sort_col='ophys_session_id')
        obtained = pd.read_csv(os.path.join(tmp_dir,
                                            'ophys_session_table.csv'),
                               dtype={'mouse_id': str},
                               parse_dates=['date_of_acquisition'])
        obtained = sort_df(df=obtained, sort_col='ophys_session_id')
        convert_strings_to_lists(df=obtained)
        pd.testing.assert_frame_equal(expected,
                                      obtained)

        # test ophys experiment
        expected = pd.read_pickle(os.path.join(expected_path,
                                               'ophys_experiment_table.pkl'))
        expected = sort_df(df=expected, sort_col='ophys_experiment_id')
        obtained = pd.read_csv(os.path.join(tmp_dir,
                                            'ophys_experiment_table.csv'),
                               dtype={'mouse_id': str},
                               parse_dates=['date_of_acquisition'])
        obtained = sort_df(df=obtained, sort_col='ophys_experiment_id')
        convert_strings_to_lists(df=obtained, is_session=False)
        pd.testing.assert_frame_equal(expected,
                                      obtained)
