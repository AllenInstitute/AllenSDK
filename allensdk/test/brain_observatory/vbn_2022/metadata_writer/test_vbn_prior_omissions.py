# Since the VBN prior_exposures_to_omissions column needs to
# take into account two histories (behavior sessions and
# ecephys sessions), we are splitting out tests of
# _add_prior_exposures_to_omissions into a separate
# module so that we can rigorously define a dataset
# with reasonably self-contained pytest fixtures.

import pytest
import pandas as pd
import numpy as np
import datetime
import copy

from allensdk.brain_observatory.vbn_2022 \
    .metadata_writer.dataframe_manipulations import (
        _add_prior_omissions)


@pytest.fixture
def session_list_fixture():
    """
    Return a list of dicts. Each dict will contain
        mouse_id
        behavior_session_id
        ecephys_session_id
        date_of_acquisition
        session_type
        prior_exposure_to_omissions expected at this point for this mouse
    """

    rng = np.random.default_rng(22221111)
    output = []

    # session_type followed by boolean indicating
    # if it should increment prior_exposures_to_omissions
    session_types = [('junk_ephys', True),
                     ('silly', False),
                     ('ephys_junk', True),
                     ('nonsense', False),
                     ('some_ecephys_data', True),
                     (None, False)]

    tt = 12345678.1
    beh = 0
    ece = 100
    for mouse_id in (1, 2, 3):
        ct_omissions = 0
        n_sessions = rng.integers(20, 37)
        for ii in range(n_sessions):
            beh += 1
            ece += 1
            session_id_type = rng.choice((0, 1, 2))
            if session_id_type == 0:
                behavior_id = beh
                ecephys_id = ece
            elif session_id_type == 1:
                behavior_id = None
                ecephys_id = ece
            elif session_id_type == 2:
                behavior_id = beh
                ecephys_id = None

            session_type = rng.choice(session_types)

            tt += rng.random()*10.0
            if session_type[0] is None:
                prior = None
            else:
                prior = ct_omissions

            this_session = {
                'mouse_id': mouse_id,
                'behavior_session_id': behavior_id,
                'ecephys_session_id': ecephys_id,
                'date_of_acquisition': datetime.datetime.fromtimestamp(tt),
                'session_type': session_type[0],
                'prior_exposures_to_omissions': prior}
            output.append(this_session)
            if session_type[1]:
                ct_omissions += 1

    return output


@pytest.fixture
def session_dfs_fixture(
        session_list_fixture):
    """
    Return behavior_sessions_df and ecephys_sessions_df
    based on session_list_fixture. Include both input dataframes
    (those without 'prior_exposures_to_omissions') and expected
    dataframes (those with 'prior_exposures_to_omissions')
    """
    session_list = copy.deepcopy(session_list_fixture)

    rng = np.random.default_rng(2231231)
    behavior_data = []
    ecephys_data = []
    for session in session_list:
        if session['ecephys_session_id'] is None:
            is_beh = True
            is_ece = False
        elif session['behavior_session_id'] is None:
            is_beh = False
            is_ece = True
        else:
            is_beh = rng.choice((True, False))
            is_ece = rng.choice((True, False))

        if not is_beh and not is_ece:
            is_beh = True

        if is_beh:
            behavior_data.append(session)

        if is_ece:
            ecephys_data.append(session)

    assert len(behavior_data) > 0
    assert len(ecephys_data) > 0

    expected_behavior_df = pd.DataFrame(data=behavior_data)
    expected_ecephys_df = pd.DataFrame(data=ecephys_data)

    input_behavior_df = expected_behavior_df.copy().drop(
            axis='columns', columns=['prior_exposures_to_omissions'])
    input_ecephys_df = expected_ecephys_df.copy().drop(
            axis='columns', columns=['prior_exposures_to_omissions'])

    return {'behavior': input_behavior_df,
            'ecephys': input_ecephys_df,
            'expected_behavior': expected_behavior_df,
            'expected_ecephys': expected_ecephys_df}


def test_vbn_add_prior_omissions(
        session_dfs_fixture):
    """
    test _add_prior_omissions on actual data
    """
    behavior_df = session_dfs_fixture['behavior'].copy(deep=True)
    ecephys_df = session_dfs_fixture['ecephys'].copy(deep=True)

    for df in (behavior_df, ecephys_df):
        assert 'prior_exposures_to_omissions' not in df.columns

    result = _add_prior_omissions(
                behavior_sessions_df=behavior_df,
                ecephys_sessions_df=ecephys_df)

    behavior_df = result['behavior']
    ecephys_df = result['ecephys']
    for df in (behavior_df, ecephys_df):
        assert 'prior_exposures_to_omissions' in df.columns

    pd.testing.assert_frame_equal(
        behavior_df,
        session_dfs_fixture['expected_behavior'],
        check_dtype=False)

    pd.testing.assert_frame_equal(
        ecephys_df,
        session_dfs_fixture['expected_ecephys'],
        check_dtype=False)


def test_vbn_date_off_warning(
        session_dfs_fixture):
    """
    Test that a warning is raised when behavior_sessions_df
    and ecephys_sessions_df disagree on the date of a session
    """

    behavior_sessions = session_dfs_fixture['behavior'].copy(deep=True)
    ecephys_sessions = session_dfs_fixture['ecephys'].copy(deep=True)

    in_beh = set([beh for beh in
                  behavior_sessions.loc[
                      behavior_sessions.behavior_session_id.notnull()
                  ].behavior_session_id])

    in_ece = set(ecephys_sessions.behavior_session_id)
    in_both = in_ece.intersection(in_beh)
    assert len(in_both) > 0
    in_both = list(in_both)
    in_both.sort()
    chosen_id = in_both[len(in_both)//2]

    beh_date = behavior_sessions.loc[
        behavior_sessions.behavior_session_id == chosen_id]
    ece_date = ecephys_sessions.loc[
        ecephys_sessions.behavior_session_id == chosen_id]

    # make sure the two dataframes agree on the date initially
    assert len(beh_date) == 1
    assert len(ece_date) == 1
    beh_date = beh_date.date_of_acquisition.values[0]
    ece_date = ece_date.date_of_acquisition.values[0]
    assert beh_date == ece_date

    # change the date in one of the frames and make sure the
    # disagreement shows up
    behavior_sessions.loc[
        behavior_sessions.behavior_session_id == chosen_id,
        'date_of_acquisition'] = datetime.datetime.fromtimestamp(99.0)

    beh_date = behavior_sessions.loc[
        behavior_sessions.behavior_session_id == chosen_id]
    ece_date = ecephys_sessions.loc[
        ecephys_sessions.behavior_session_id == chosen_id]

    assert len(beh_date) == 1
    assert len(ece_date) == 1
    beh_date = beh_date.date_of_acquisition.values[0]
    ece_date = ece_date.date_of_acquisition.values[0]
    assert beh_date != ece_date

    with pytest.warns(UserWarning,
                      match="disagree on the date of behavior session"):
        _add_prior_omissions(
                behavior_sessions_df=behavior_sessions,
                ecephys_sessions_df=ecephys_sessions)


def test_vbn_session_type_off_error(
        session_dfs_fixture):
    """
    Test that an exception is thrown when behavior_sessions_df
    and ecephys_sessions_df disagree on the session_type
    """

    behavior_sessions = session_dfs_fixture['behavior'].copy(deep=True)
    ecephys_sessions = session_dfs_fixture['ecephys'].copy(deep=True)

    in_beh = set([beh for beh in
                  behavior_sessions.loc[
                      behavior_sessions.session_type.notnull()
                      & behavior_sessions.behavior_session_id.notnull()
                  ].behavior_session_id])

    in_ece = set(ecephys_sessions.behavior_session_id)

    in_both = in_ece.intersection(in_beh)
    assert len(in_both) > 0
    in_both = list(in_both)
    in_both.sort()
    chosen_id = in_both[len(in_both)//2]

    beh_type = behavior_sessions.loc[
        behavior_sessions.behavior_session_id == chosen_id]
    ece_type = ecephys_sessions.loc[
        ecephys_sessions.behavior_session_id == chosen_id]

    # make sure the two dataframes agree on the session_type initially
    assert len(beh_type) == 1
    assert len(ece_type) == 1
    beh_type = beh_type.session_type.values[0]
    ece_type = ece_type.session_type.values[0]
    assert beh_type == ece_type

    # change the session_type in one of the frames and make sure the
    # disagreement shows up
    behavior_sessions.loc[
        behavior_sessions.behavior_session_id == chosen_id,
        'session_type'] = 'something_new_and_bizarre'

    beh_type = behavior_sessions.loc[
        behavior_sessions.behavior_session_id == chosen_id]
    ece_type = ecephys_sessions.loc[
        ecephys_sessions.behavior_session_id == chosen_id]

    assert len(beh_type) == 1
    assert len(ece_type) == 1
    beh_type = beh_type.session_type.values[0]
    ece_type = ece_type.session_type.values[0]
    assert beh_type != ece_type

    with pytest.raises(RuntimeError,
                       match="disagree on the session type"):
        _add_prior_omissions(
                behavior_sessions_df=behavior_sessions,
                ecephys_sessions_df=ecephys_sessions)
