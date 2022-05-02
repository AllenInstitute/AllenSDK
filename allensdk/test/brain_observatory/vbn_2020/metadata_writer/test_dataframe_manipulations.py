import pandas as pd

from allensdk.brain_observatory.vbn_2022.metadata_writer \
    .dataframe_manipulations import (
        _add_session_number,
        _add_prior_omissions_to_ecephys,
        _add_prior_omissions_to_behavior)


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


def test_add_prior_omissions_to_ecephys():
    """
    Test that _add_prior_omissions_to_ecephys just
    adds a column that is session_number-1
    """

    input_data = []
    input_data.append({'a': 2, 'session_number': 4})
    input_data.append({'a': 5, 'session_number': 13})
    input_df = pd.DataFrame(data=input_data)
    actual = _add_prior_omissions_to_ecephys(sessions_df=input_df)

    for session in input_data:
        n = session['session_number']
        session['prior_exposures_to_omissions'] = n - 1

    expected = pd.DataFrame(data=input_data)
    pd.testing.assert_frame_equal(expected, actual)


def test_add_prior_omissions_to_behavior():
    """
    Test that _add_prior_omissions_to_behavior gets the right
    relationship between ecephys and behavior sessions
    """

    ecephys_data = []
    ecephys_data.append(
        {'mouse_id': '1',
         'ecephys_session_id': 0,
         'date_of_acquisition': pd.Timestamp(2020, 7, 11)})
    ecephys_data.append(
        {'mouse_id': '1',
         'ecephys_session_id': 1,
         'date_of_acquisition': pd.Timestamp(2020, 8, 12)})
    ecephys_data.append(
        {'mouse_id': '1',
         'ecephys_session_id': 5,
         'date_of_acquisition': pd.Timestamp(2020, 11, 12)})

    ecephys_data.append(
        {'mouse_id': '2',
         'ecephys_session_id': 3,
         'date_of_acquisition': pd.Timestamp(2020, 9, 4)})
    ecephys_data.append(
        {'mouse_id': '2',
         'ecephys_session_id': 4,
         'date_of_acquisition': pd.Timestamp(2020, 10, 2)})

    ecephys_df = pd.DataFrame(data=ecephys_data)
    ecephys_df = _add_session_number(
                    sessions_df=ecephys_df,
                    index_col='ecephys_session_id')
    ecephys_df = _add_prior_omissions_to_ecephys(
                    sessions_df=ecephys_df)

    behavior_data = []
    behavior_data.append(
        {'mouse_id': '1',
         'behavior_session_id': 0,
         'date_of_acquisition': pd.Timestamp(2020, 5, 3)})
    behavior_data.append(
        {'mouse_id': '1',
         'behavior_session_id': 1,
         'date_of_acquisition': pd.Timestamp(2020, 7, 14)})
    behavior_data.append(
        {'mouse_id': '1',
         'behavior_session_id': 2,
         'date_of_acquisition': pd.Timestamp(2020, 8, 12)})
    behavior_data.append(
        {'mouse_id': '1',
         'behavior_session_id': 3,
         'date_of_acquisition': pd.Timestamp(2020, 8, 13)})
    behavior_data.append(
        {'mouse_id': '1',
         'behavior_session_id': 4,
         'date_of_acquisition': pd.Timestamp(2020, 11, 12)})
    behavior_data.append(
        {'mouse_id': '1',
         'behavior_session_id': 5,
         'date_of_acquisition': pd.Timestamp(2020, 11, 14)})

    behavior_data.append(
        {'mouse_id': '2',
         'behavior_session_id': 6,
         'date_of_acquisition': pd.Timestamp(2020, 9, 3)})
    behavior_data.append(
        {'mouse_id': '2',
         'behavior_session_id': 7,
         'date_of_acquisition': pd.Timestamp(2020, 9, 4)})
    behavior_data.append(
        {'mouse_id': '2',
         'behavior_session_id': 8,
         'date_of_acquisition': pd.Timestamp(2020, 10, 2)})
    behavior_data.append(
        {'mouse_id': '2',
         'behavior_session_id': 9,
         'date_of_acquisition': pd.Timestamp(2020, 10, 3)})

    behavior_df = pd.DataFrame(data=behavior_data)
    behavior_df = _add_prior_omissions_to_behavior(
                    behavior_df=behavior_df,
                    ecephys_df=ecephys_df)

    behavior_data[0]['prior_exposures_to_omissions'] = 0
    behavior_data[1]['prior_exposures_to_omissions'] = 1
    behavior_data[2]['prior_exposures_to_omissions'] = 1
    behavior_data[3]['prior_exposures_to_omissions'] = 2
    behavior_data[4]['prior_exposures_to_omissions'] = 2
    behavior_data[5]['prior_exposures_to_omissions'] = 3

    behavior_data[6]['prior_exposures_to_omissions'] = 0
    behavior_data[7]['prior_exposures_to_omissions'] = 0
    behavior_data[8]['prior_exposures_to_omissions'] = 1
    behavior_data[9]['prior_exposures_to_omissions'] = 2

    expected = pd.DataFrame(data=behavior_data)
    pd.testing.assert_frame_equal(behavior_df, expected)
