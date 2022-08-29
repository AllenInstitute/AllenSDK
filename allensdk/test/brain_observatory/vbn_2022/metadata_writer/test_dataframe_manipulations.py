import pytest
import pandas as pd
import datetime
import numpy as np
import copy

from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.vbn_2022.metadata_writer \
    .dataframe_manipulations import (
        _add_session_number,
        _add_experience_level,
        _patch_date_and_stage_from_pickle_file,
        _add_age_in_days,
        _add_images_from_behavior,
        strip_substructure_acronym_df)
from allensdk.internal.brain_observatory.util.multi_session_utils import \
    remove_invalid_sessions

from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


def test_add_images_from_behavior():
    """
    Test that _add_images_from_behavior
    maps the right values from the behavior session
    table to the ecephys_session_table
    """
    beh_data = []
    beh_data.append({'ecephys_session_id': 1,
                     'image_set': 'zebra',
                     'prior_exposures_to_image_set': 22})

    beh_data.append({'ecephys_session_id': 2,
                     'image_set': 'fish',
                     'prior_exposures_to_image_set': 14})

    beh_data.append({'ecephys_session_id': None,
                     'image_set': 'lion',
                     'prior_exposures_to_image_set': 5})

    beh_data.append({'ecephys_session_id': 7,
                     'image_set': 'tiger',
                     'prior_exposures_to_image_set': 4})

    beh_table = pd.DataFrame(data=beh_data)

    ecephys_data = []
    ecephys_data.append({'ecephys_session_id': 2,
                         'figure': 3})
    ecephys_data.append({'ecephys_session_id': 1,
                         'figure': 7})
    ecephys_data.append({'ecephys_session_id': 3,
                         'figure': 8})

    ecephys_table = pd.DataFrame(data=ecephys_data)
    ecephys_table = _add_images_from_behavior(
                ecephys_table=ecephys_table,
                behavior_table=beh_table)

    expected_data = []
    expected_data.append({'ecephys_session_id': 2,
                          'figure': 3,
                          'image_set': 'fish',
                          'prior_exposures_to_image_set': 14})
    expected_data.append({'ecephys_session_id': 1,
                          'figure': 7,
                          'image_set': 'zebra',
                          'prior_exposures_to_image_set': 22})
    expected_data.append({'ecephys_session_id': 3,
                          'figure': 8,
                          'image_set': None,
                          'prior_exposures_to_image_set': None})

    expected_df = pd.DataFrame(data=expected_data)
    pd.testing.assert_frame_equal(ecephys_table, expected_df)


def test_add_session_number():
    """
    Test that the session_number column added by
    _add_session_number is correct
    """

    input_data = []
    input_data.append(
        {'mouse_id': '1',
         'session_id': 5,
         'date_of_acquisition': pd.Timestamp(2021, 7, 18)})
    input_data.append(
        {'mouse_id': '2',
         'session_id': 3,
         'date_of_acquisition': pd.Timestamp(1999, 6, 11)})
    input_data.append(
        {'mouse_id': '1',
         'session_id': 1,
         'date_of_acquisition': pd.Timestamp(2021, 6, 5)})
    input_data.append(
        {'mouse_id': '1',
         'session_id': 6,
         'date_of_acquisition': pd.Timestamp(2022, 5, 6)})
    input_data.append(
        {'mouse_id': '2',
         'session_id': 2,
         'date_of_acquisition': pd.Timestamp(2001, 3, 4)})
    input_data.append(
        {'mouse_id': '3',
         'session_id': 4,
         'date_of_acquisition': pd.Timestamp(2001, 3, 4)})

    input_df = pd.DataFrame(data=input_data)
    actual = _add_session_number(sessions_df=input_df,
                                 index_col='session_id')

    # construct expectd output
    input_data[0]['session_number'] = 2
    input_data[2]['session_number'] = 1
    input_data[3]['session_number'] = 3

    input_data[1]['session_number'] = 1
    input_data[4]['session_number'] = 2

    input_data[5]['session_number'] = 1

    expected = pd.DataFrame(data=input_data)

    pd.testing.assert_frame_equal(expected, actual)


def test_add_experience_level():
    """
    Test that _add_experience level behaves properly
    """

    input_data = []
    input_data.append({'session': 1,
                       'prior_exposures_to_image_set': 2})

    input_data.append({'session': 2,
                       'prior_exposures_to_image_set': None})

    input_data.append({'session': 1,
                       'prior_exposures_to_image_set': 0})

    input_df = pd.DataFrame(data=input_data)
    actual = _add_experience_level(sessions_df=input_df)

    input_data[0]['experience_level'] = 'Familiar'
    input_data[1]['experience_level'] = 'Novel'
    input_data[2]['experience_level'] = 'Novel'

    expected = pd.DataFrame(data=input_data)
    pd.testing.assert_frame_equal(actual, expected)
    assert 'experience_level' in actual.columns


@pytest.mark.parametrize(
        "flag_columns, ids_to_fix, cols_to_fix",
        [(['date_of_acquisition', 'foraging_id', 'session_type'],
          (1123, 5813, 2134),
          ['date_of_acquisition', ]),
         (['date_of_acquisition', 'foraging_id', 'session_type'],
          (1123, 5813, 2134),
          ['date_of_acquisition', 'session_type']),
         (['date_of_acquisition', 'foraging_id', 'session_type'],
          (1123, 5813, 2134),
          ['session_type']),
         (['date_of_acquisition', 'foraging_id', 'session_type'],
          (1123, 5813, 2134),
          None),
         (['foraging_id', 'session_type'],
          (2134, 5813),
          None),
         (['foraging_id', 'session_type'],
          (2134, 5813),
          ['date_of_acquisition', 'session_type']),
         (['foraging_id', 'session_type'],
          (2134, 5813),
          ['date_of_acquisition', ]),
         (['foraging_id', 'session_type'],
          (2134, 5813),
          ['session_type', ]),
         (['foraging_id', ], (2134, ), None),
         (['foraging_id', ], (2134, ),
          ['date_of_acquisition', 'session_type']),
         (['foraging_id', ], (2134, ), ['date_of_acquisition', ]),
         (['foraging_id', ], (2134, ), ['session_type', ])])
def test_patch_date_and_stage_from_pickle_file(
        patching_pickle_file_fixture,
        flag_columns,
        ids_to_fix,
        cols_to_fix):
    """
    Test that _patch_date_and_stage_from_pickle_file
    correctly patches sessions that are missing
    date_of_acquisition or session_type

    flag_columns is passed along to
    _patch_date_and_stage_from_pickle_file

    ids_to_fix denotes the behavior_session_id of the sessions
    we expect to be patched

    cols_to_fix denotes the columns we want patched
    """

    input_data = []
    input_data.append(
        {'behavior_session_id': 213455,
         'date_of_acquisition': datetime.datetime(1982, 11, 13),
         'session_type': 'perfection_reasonable',
         'foraging_id': 2222,
         'silly': 'foo'})

    input_data.append(
        {'behavior_session_id': 1123,
         'date_of_acquisition': None,
         'session_type': 'perfection_reasonable',
         'foraging_id': 1111,
         'silly': 'bar'})

    input_data.append(
        {'behavior_session_id': 5813,
         'date_of_acquisition': datetime.datetime(1982, 11, 13),
         'session_type': None,
         'foraging_id': 3333,
         'silly': 'baz'})

    input_data.append(
        {'behavior_session_id': 2134,
         'date_of_acquisition': datetime.datetime(1982, 11, 13),
         'session_type': 4444,
         'foraging_id': None,
         'silly': 'baz'})

    input_df = pd.DataFrame(data=input_data)

    class DummyLimsConnection(object):

        @staticmethod
        def select(query):
            df_data = []

            # get session_id_list from last line of query
            last_line = query.strip().split('\n')[-1]
            last_line = last_line.split('(')[-1]
            last_line = last_line.replace(')', '')
            last_line = [x.strip() for x in last_line.split(',')]
            behavior_session_id_list = [
                int(ii) for ii in last_line]

            for bid in behavior_session_id_list:
                pkl_path = patching_pickle_file_fixture[bid]
                pkl_path = pkl_path['pkl_path']
                pkl_path = str(pkl_path.resolve().absolute())
                element = {
                    'behavior_session_id': bid,
                    'pkl_path': pkl_path}
                df_data.append(element)
            return pd.DataFrame(data=df_data)

    actual = _patch_date_and_stage_from_pickle_file(
                    lims_connection=DummyLimsConnection,
                    behavior_df=input_df,
                    flag_columns=flag_columns,
                    columns_to_patch=cols_to_fix)

    if cols_to_fix is None:
        cols_to_fix = ['date_of_acquisition', 'session_type']

    expected_data = copy.deepcopy(input_data)
    for element in expected_data:
        bid = element['behavior_session_id']
        if bid not in ids_to_fix:
            continue
        for col in cols_to_fix:
            this_s = patching_pickle_file_fixture[bid][col]
            element[col] = this_s

    expected = pd.DataFrame(data=expected_data)
    pd.testing.assert_frame_equal(actual,
                                  expected)


def test_patch_date_and_stage_from_pickle_file_error():
    """
    Make sure that an error gets raised when you give
    _patch_date_and_stage_from_pickle_file an unexpected
    column to patch
    """
    with pytest.raises(ValueError, match="can only patch"):
        _patch_date_and_stage_from_pickle_file(
            lims_connection=None,
            behavior_df=None,
            flag_columns=['date_of_acquisition', ],
            columns_to_patch=['date_of_acquisition', 'flavor'])


@pytest.mark.parametrize('index_column', ['behavior_session_id', 'other_id'])
def test_add_age_in_days(index_column):
    """
    Test that _add_age_in_days adds the expected column to a dataframe
    """
    expected_age = []
    input_data = []

    input_data.append(
       {index_column: 1,
        'date_of_acquisition': datetime.datetime(2020, 7, 8, 12),
        'date_of_birth': datetime.datetime(2020, 7, 6, 14)})
    expected_age.append(2)

    input_data.append(
       {index_column: 2,
        'date_of_acquisition': datetime.datetime(2020, 7, 9, 15),
        'date_of_birth': datetime.datetime(2020, 7, 6, 14)})
    expected_age.append(3)

    input_df = pd.DataFrame(data=input_data)
    actual = _add_age_in_days(df=input_df,
                              index_column=index_column)

    for session, age in zip(input_data, expected_age):
        session['age_in_days'] = age
    expected = pd.DataFrame(data=input_data)
    assert 'age_in_days' in actual.columns
    pd.testing.assert_frame_equal(actual, expected)


def test_remove_pretest_sessions():
    """
    Test that remove_pretest_sessions only removes
    sessions whose session_types starts with 'pretest_'
    """

    raw_data = [
        {'id': 0, 'session_type': 'A', 'col': 'foo'},
        {'id': 1, 'session_type': 'pretest_B', 'col': 'bar'},
        {'id': 2, 'session_type': 'C_pretest', 'col': 'baz'},
        {'id': 3, 'session_type': 'D', 'col': 'pretest'}]
    behavior_sessions = [
        BehaviorMetadata(
            date_of_acquisition=None,
            subject_metadata=None,
            behavior_session_id=None,
            equipment=None,
            stimulus_frame_rate=None,
            session_type=SessionType(x['session_type']),
            behavior_session_uuid=None,
            session_duration=None
        )
        for x in raw_data]
    actual = remove_invalid_sessions(
        behavior_sessions=behavior_sessions,
        remove_pretest_sessions=True,
        remove_aborted_sessions=False,
        remove_sessions_after_mouse_death_date=False)

    expected = [
        behavior_sessions[0],
        behavior_sessions[2],
        behavior_sessions[3]
    ]
    assert actual == expected


@pytest.mark.parametrize(
    "input_data, output_data, col_name",
    [([{'a': 1, 'b': 'DG-mo'},
       {'a': 2, 'b': 'LS-x'},
       {'a': 3, 'b': None},
       {'a': 4, 'b': [None, ]},
       {'a': 5, 'b': np.NaN}],
      [{'a': 1, 'b': 'DG'},
       {'a': 2, 'b': 'LS'},
       {'a': 3, 'b': None},
       {'a': 4, 'b': []},
       {'a': 5, 'b': None}],
      'b'),
     ([{'a': 1, 'b': ['DG-mo', 'AB-x', 'DG-pb', np.NaN]},
       {'a': 2, 'b': 'DG-s'}],
      [{'a': 1, 'b': ['AB', 'DG']},
       {'a': 2, 'b': 'DG'}],
      'b')])
def test_strip_substructure_acronym_df(
        input_data,
        output_data,
        col_name):
    """
    Test method that strips the substructure acronym
    columns in a dataframe
    """

    input_df = pd.DataFrame(data=input_data)
    expected_df = pd.DataFrame(data=output_data)

    actual_df = strip_substructure_acronym_df(
            df=input_df,
            col_name=col_name)

    pd.testing.assert_frame_equal(expected_df, actual_df)


class TestDataframeManipulations(LimsTest):
    @classmethod
    def setup_class(cls):
        cls.behavior_sessions_df = \
            pd.DataFrame(
                [
                    {'behavior_session_id': 1,
                     'session_type': 'EPHYS_1_images_G_5uL_reward'},
                    {'behavior_session_id': 2,
                     'session_type': 'EPHYS_1_images_G_5uL_reward'},
                    {'behavior_session_id': 3,
                     'session_type':
                         'TRAINING_0_gratings_autorewards_15min_0uL_reward'
                     },
                    {'behavior_session_id': 4,
                     'session_type':
                         'TRAINING_0_gratings_autorewards_15min_0uL_reward'
                     }
                ]
            ).set_index('behavior_session_id')
        session_durations = \
            pd.Series({
                # Too short
                1: 3599,
                # Valid
                2: 3601,
                # Too short
                3: 899,
                # Valid
                4: 901
            })
        session_durations.index.name = 'behavior_session_id'
        cls.session_durations = session_durations

    def test_remove_aborted_sessions(self):
        behavior_sessions = [
            BehaviorMetadata(
                date_of_acquisition=None,
                subject_metadata=None,
                behavior_session_id=BehaviorSessionId(x),
                equipment=None,
                stimulus_frame_rate=None,
                session_type=(
                    SessionType(
                        self.behavior_sessions_df.loc[x]['session_type'])),
                behavior_session_uuid=None,
                session_duration=self.session_durations.loc[x]
            )
            for x in self.behavior_sessions_df.index]
        actual = remove_invalid_sessions(
            behavior_sessions=behavior_sessions,
            remove_pretest_sessions=False,
            remove_aborted_sessions=True,
            remove_sessions_after_mouse_death_date=False)
        expected = [
            behavior_sessions[1],
            behavior_sessions[3]
        ]
        assert \
            sorted([x.behavior_session_id for x in actual]) == \
            sorted([x.behavior_session_id for x in expected])
