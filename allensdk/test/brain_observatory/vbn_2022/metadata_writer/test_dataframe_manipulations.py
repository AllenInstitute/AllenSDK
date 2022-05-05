import pytest
import pandas as pd
import datetime
import copy

from allensdk.brain_observatory.vbn_2022.metadata_writer \
    .id_generator import (
        FileIDGenerator)

from allensdk.brain_observatory.vbn_2022.metadata_writer \
    .dataframe_manipulations import (
        _add_session_number,
        _add_experience_level,
        _patch_date_and_stage_from_pickle_file,
        _add_age_in_days,
        _add_images_from_behavior,
        add_file_paths_to_session_table)


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


def test_patch_date_and_stage_from_pickle_file(
        patching_pickle_file_fixture):
    """
    Test that _patch_date_and_stage_from_pickle_file
    correctly patches sessions that are missing
    date_of_acquisition or session_type
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
            last_line = last_line.replace(',', '').replace(')', '')
            last_line = last_line.strip().split()
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
                    behavior_df=input_df)

    expected_data = copy.deepcopy(input_data)
    for idx, bid in enumerate((1123, 5813)):
        this_d = patching_pickle_file_fixture[bid]['date_of_acquisition']
        this_s = patching_pickle_file_fixture[bid]['session_type']
        expected_data[idx+1]['date_of_acquisition'] = this_d
        expected_data[idx+1]['session_type'] = this_s

    expected = pd.DataFrame(data=expected_data)
    pd.testing.assert_frame_equal(actual, expected)


def test_add_age_in_days():
    """
    Test that _add_age_in_days adds the expected column to a dataframe
    """
    expected_age = []
    input_data = []

    input_data.append(
       {'behavior_session_id': 1,
        'date_of_acquisition': datetime.datetime(2020, 7, 8, 12),
        'date_of_birth': datetime.datetime(2020, 7, 6, 14)})
    expected_age.append(1)

    input_data.append(
       {'behavior_session_id': 2,
        'date_of_acquisition': datetime.datetime(2020, 7, 9, 15),
        'date_of_birth': datetime.datetime(2020, 7, 6, 14)})
    expected_age.append(3)

    input_df = pd.DataFrame(data=input_data)
    actual = _add_age_in_days(df=input_df)

    for session, age in zip(input_data, expected_age):
        session['age_in_days'] = age
    expected = pd.DataFrame(data=input_data)
    assert 'age_in_days' in actual.columns
    pd.testing.assert_frame_equal(actual, expected)


def test_add_file_paths_to_session_table_on_missing_error(
        some_files_fixture,
        session_table_fixture):
    """
    Test that an error is raised by add_file_paths_to_session_table
    when on_missing_file is a nonsense value
    """
    with pytest.raises(ValueError,
                       match="on_missing_file must be one of"):
        add_file_paths_to_session_table(
            session_table=session_table_fixture,
            id_generator=FileIDGenerator(),
            file_dir=some_files_fixture[0].parent,
            file_prefix='silly_file',
            index_col='file_index',
            on_missing_file='whatever')


def test_add_file_paths_to_session_table_no_file_error(
        some_files_fixture,
        session_table_fixture):
    """
    Test that an error is raised by add_file_paths_to_session_table
    when files are missing (if requested)
    """
    with pytest.raises(RuntimeError,
                       match="The following files do not exist"):
        add_file_paths_to_session_table(
            session_table=session_table_fixture,
            id_generator=FileIDGenerator(),
            file_dir=some_files_fixture[0].parent,
            file_prefix='silly_file',
            index_col='file_index',
            on_missing_file='error')


@pytest.mark.parametrize(
        'on_missing_file', ['skip', 'warn'])
def test_add_file_paths_to_session_table(
        some_files_fixture,
        session_table_fixture,
        on_missing_file):
    """
    Test that add_file_paths_to_session_table behaves as expected
    when not raising an error
    """
    file_dir = some_files_fixture[0].parent
    expected = session_table_fixture.copy(deep=True)

    id_generator = FileIDGenerator()

    with pytest.warns(UserWarning,
                      match='The following files do not exist'):
        result = add_file_paths_to_session_table(
                    session_table=session_table_fixture,
                    id_generator=id_generator,
                    file_dir=file_dir,
                    file_prefix='silly_file',
                    index_col='file_index',
                    on_missing_file=on_missing_file)

    # because we have not yet added file_id and file_path
    # to expected
    assert not expected.equals(result)

    if on_missing_file == 'skip':
        assert len(result) == len(some_files_fixture)
    else:
        assert len(result) == len(some_files_fixture) + 2

    expected['file_id'] = id_generator.dummy_value
    expected['file_path'] = 'nothing'

    for file_idx in expected.file_index:
        file_path = file_dir / f'silly_file_{file_idx}.nwb'
        str_path = str(file_path.resolve().absolute())
        expected.loc[expected.file_index == file_idx, 'file_path'] = str_path
        if file_path.exists():
            file_id = id_generator.id_from_path(file_path=file_path)
            expected.loc[expected.file_index == file_idx, 'file_id'] = file_id
        elif on_missing_file == 'skip':
            expected = expected.drop(
                    expected.loc[expected.file_index == file_idx].index)

    pd.testing.assert_frame_equal(result, expected)
