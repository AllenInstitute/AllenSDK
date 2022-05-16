# This module contains utility methods for extending
# the VBN 2022 metadata dataframes as they are directly queried
# from LIMS.

from typing import Dict, List
import pandas as pd
import numpy as np
import json
import warnings

from allensdk.internal.api import PostgresQueryMixin

from allensdk.brain_observatory.behavior.behavior_project_cache \
    .tables.util.prior_exposure_processing import (
        __get_prior_exposure_count)

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile)

from allensdk.internal.api.queries.behavior_lims_queries import (
    stimulus_pickle_paths_from_behavior_session_ids)


def _add_session_number(
        sessions_df: pd.DataFrame,
        index_col: str) -> pd.DataFrame:
    """
    For each mouse: order sessions by date_of_acquisition. Add a session_number
    column corresponding to where that session falls in the mouse's history.

    Parameters
    ----------
        sessions_df: pd.DataFrame

        index_col: str
            The column denoting the unique ID of each
            session. Should be either "ecephys_session_id"
            or "behavior_session_id"

    Returns
    -------
    sessions_df: pd.DataFrame
        The input dataframe with a session_number column added

    Note
    ----
    session_number will be 1-indexed
    """

    date_col = 'date_of_acquisition'
    mouse_col = 'mouse_id'

    mouse_id_values = np.unique(sessions_df[mouse_col].values)
    new_data = []
    for mouse_id in mouse_id_values:
        sub_df = sessions_df.query(f"{mouse_col}=='{mouse_id}'")
        sub_df = json.loads(sub_df.to_json(orient='index'))
        session_arr = []
        date_arr = []
        for index_val in sub_df.keys():
            session_arr.append(sub_df[index_val][index_col])
            date_arr.append(sub_df[index_val][date_col])
        session_arr = np.array(session_arr)
        date_arr = np.array(date_arr)
        sorted_dex = np.argsort(date_arr)
        session_arr = session_arr[sorted_dex]
        for session_number, session_id in enumerate(session_arr):
            element = {index_col: session_id,
                       'session_number': session_number+1}
            new_data.append(element)
    new_df = pd.DataFrame(data=new_data)

    sessions_df = sessions_df.join(
                        new_df.set_index(index_col),
                        on=index_col,
                        how='left')
    return sessions_df


def _add_prior_omissions(
        behavior_sessions_df: pd.DataFrame,
        ecephys_sessions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Add the 'prior_exposures_to_omissions' column assuming that
    only sessions with 'EPHYS' in the session_type included
    omissions in them.

    Because each mouse's history could be split up between
    the behavior sessions table and the ecephys sessions table,
    we need to combine the two data frames into a single history
    for each mouse, determine what each mouse has seen, and then
    set the prior_exposures_to_omissions in each dataframe
    in such a way that the ecephys sessions table knows about
    what the mouse saw in the behavior sessions table.

    This should be one of the last processing steps, as it
    depends on session_type being properly set in the
    dataframes.

    Parameters
    ----------
    behavior_sessions_df: pd.DataFrame
        the table of behavior sessions

    ecephys_sessions_df: pd.DataFrame
        the table of ecephys sessions (may or may not be
        a superset of behavior sessions)

    Returns
    -------
    updated_tables: Dict[str, pd.DataFrame]
        {'behavior': behavior_session_df with added column
         'ecephys': ecephys_sessions_df with added column}
    """

    if 'behavior_session_id' not in ecephys_sessions_df.columns:
        raise RuntimeError(
            "Cannot properly merge behavior_sessions_df and "
            "ecephys_sessions_df; ecephys_sessions_df does not have "
            "a behavior_session_id column")

    # get all of the behavior sessions
    beh_history_lookup = dict()
    for mouse_id, beh_id, date_acq, session_type in zip(
                behavior_sessions_df.mouse_id,
                behavior_sessions_df.behavior_session_id,
                behavior_sessions_df.date_of_acquisition,
                behavior_sessions_df.session_type):
        element = {'mouse_id': mouse_id,
                   'behavior_session_id': beh_id,
                   'date_of_acquisition': date_acq,
                   'ecephys_session_id': None,
                   'session_type': session_type}
        beh_history_lookup[beh_id] = element

    # get any ecephys sessions that did not occur in the
    # behavior sessions table
    full_history = []
    for mouse_id, beh_id, ece_id, date_acq, session_type in zip(
                ecephys_sessions_df.mouse_id,
                ecephys_sessions_df.behavior_session_id,
                ecephys_sessions_df.ecephys_session_id,
                ecephys_sessions_df.date_of_acquisition,
                ecephys_sessions_df.session_type):

        if not np.isnan(beh_id):
            int_beh_id = int(beh_id)
        else:
            int_beh_id = -999

        if not np.isnan(beh_id) and int_beh_id in beh_history_lookup:
            element = beh_history_lookup[beh_id]

            # This test *should* give an error; however, there are
            # sessions in LIMS in which ecephys_sessions has a
            # date_of_acquisition and behavior_sessions does not.
            # When we patch the behavior_sessions_table from the pickle
            # file, we get a different date of acquisition than is
            # listed in the ecephys_sessions table. Until we know how
            # our stakeholders want to deal with this problem,
            # I'm going make this a warning.

            if date_acq != element['date_of_acquisition']:
                warnings.warn(
                    "behavior_sessions_df and ecephys_sessions_df "
                    "disagree on the date of behavior session "
                    f"{beh_id} (ecephys_session_id {ece_id})\n"
                    f"behavior says: {element['date_of_acquisition']}\n"
                    f"ecephys says: {date_acq}")
            if session_type != element['session_type']:
                raise RuntimeError(
                    "behavior_sessions_df and ecephys_session_df "
                    "disagree on the session type of behavior session "
                    f"{beh_id} (ecephys_session_id {ece_id})\n"
                    f"behavior says: {element['session_type']}\n"
                    f"ecephys says: {session_type}")
            element['ecephys_session_id'] = ece_id
        else:
            element = {'mouse_id': mouse_id,
                       'behavior_session_id': beh_id,
                       'ecephys_session_id': ece_id,
                       'date_of_acquisition': date_acq,
                       'session_type': session_type}
            full_history.append(element)

    # create a dataframe containing the full history (behavior and
    # ecephys sessions) of each mouse
    for beh_id in beh_history_lookup:
        full_history.append(beh_history_lookup[beh_id])

    full_history_df = pd.DataFrame(data=full_history)

    # add prior_exposures_to_omissions to the full history data frame
    contains_omissions = pd.Series(False,
                                   index=full_history_df.index)
    contains_omissions.loc[
        (full_history_df.session_type.notnull()) &
        (full_history_df.session_type.str.lower().str.contains('ephys'))
    ] = True

    full_history_df[
        'prior_exposures_to_omissions'] = __get_prior_exposure_count(
                df=full_history_df,
                to=contains_omissions,
                agg_method='cumsum')

    # merge behavior_sessions_df and ecephys_sessions_df with the
    # appropriate subsets of the full_history_df

    beh_history_df = full_history_df.loc[
                    full_history_df.behavior_session_id.notnull(),
                    ('behavior_session_id',
                     'prior_exposures_to_omissions')]

    behavior_sessions_df = behavior_sessions_df.join(
                beh_history_df.set_index('behavior_session_id'),
                on='behavior_session_id',
                how='left')

    ece_history_df = full_history_df.loc[
                        full_history_df.ecephys_session_id.notnull(),
                        ('ecephys_session_id', 'prior_exposures_to_omissions')]

    ecephys_sessions_df = ecephys_sessions_df.join(
                ece_history_df.set_index('ecephys_session_id'),
                on='ecephys_session_id',
                how='left')

    return {'behavior': behavior_sessions_df,
            'ecephys': ecephys_sessions_df}


def _add_experience_level(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the column 'experience_level' to a dataframe. This column
    will be 'Novel' for any rows with
    'prior_exposures_to_image_set' == 0 (or NULL) and 'Familiar'
    otherwise.

    Return the same dataframe with the column added
    """

    sessions_df['experience_level'] = np.where(
                  np.logical_or(
                      sessions_df['prior_exposures_to_image_set'] == 0,
                      sessions_df['prior_exposures_to_image_set'].isnull()),
                  'Novel',
                  'Familiar')
    return sessions_df


def _patch_date_and_stage_from_pickle_file(
        lims_connection: PostgresQueryMixin,
        behavior_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing date_of_acquisition and session_type
    directly from the stimulus pickle file

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    behavior_df: pd.DataFrame
        The dataframe to be patched

    Returns
    -------
    behavior_df: pd.DataFrame
        Identical to behavior_df, except any rows with NULL
        date_of_acquisition or foraging_id will have their
        date_of_acquisition and session_type overwritten with
        values from the stimulus pickle file.
    """

    invalid_beh = behavior_df[
            np.logical_or(
                behavior_df.date_of_acquisition.isna(),
                np.logical_or(
                    behavior_df.foraging_id.isna(),
                    behavior_df.session_type.isna()))
    ].behavior_session_id.values

    assert len(invalid_beh) == len(np.unique(invalid_beh))

    if len(invalid_beh) > 0:
        pickle_path_df = stimulus_pickle_paths_from_behavior_session_ids(
                            lims_connection=lims_connection,
                            behavior_session_id_list=invalid_beh.tolist())

        for beh_id, pkl_path in zip(pickle_path_df.behavior_session_id,
                                    pickle_path_df.pkl_path):
            stim_file = BehaviorStimulusFile(filepath=pkl_path)
            new_date = stim_file.date_of_acquisition
            new_session_type = stim_file.session_type

            behavior_df.loc[
                behavior_df.behavior_session_id == beh_id,
                ('date_of_acquisition', 'session_type')] = (new_date,
                                                            new_session_type)

    return behavior_df


def _add_age_in_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'age_in_days' column to a dataframe by subtracting
    'date_of_birth' from 'date_of_acquisition'. Return the
    dataframe with the new column added.
    """
    age_in_days = []
    for beh_id, acq, birth in zip(
                df.behavior_session_id,
                df.date_of_acquisition,
                df.date_of_birth):
        age = (acq-birth).days
        age_in_days.append({'behavior_session_id': beh_id,
                            'age_in_days': age})
    age_in_days = pd.DataFrame(data=age_in_days)
    df = df.join(
            age_in_days.set_index('behavior_session_id'),
            on='behavior_session_id',
            how='left')
    return df


def _add_images_from_behavior(
        ecephys_table: pd.DataFrame,
        behavior_table: pd.DataFrame) -> pd.DataFrame:
    """
    Use the behavior sessions table to add image_set and
    prior_exposures_to_image_set to ecephys table.

    Parameters
    ----------
    ecephys_table: pd.DataFrame
        A dataframe of ecephys_sessions

    behavior_table: pd.DataFrame
        A dataframe of behavior_sessions

    Returns
    -------
    ecephys_sessions:
        Same as input, except that image_set and
        prior_exposures_to_image_set have been copied
        from behavior_table where appropriate

    Notes
    -----
    Because images are more appropriately associated with
    behavior sessions, it is easiest to just assemble
    a table of behavior sessions and then join this to
    the ecephys_sessions using behavior_sessions.ecephys_session_id,
    which is effectively what this method does.
    """
    # add prior exposure to image_set to session_table

    sub_df = behavior_table.loc[
        np.logical_not(behavior_table.ecephys_session_id.isna()),
        ('ecephys_session_id', 'image_set', 'prior_exposures_to_image_set')]

    ecephys_table = ecephys_table.merge(
            sub_df.set_index('ecephys_session_id'),
            on='ecephys_session_id',
            how='left')
    return ecephys_table


def remove_aborted_sessions(
        lims_connection: PostgresQueryMixin,
        behavior_df: pd.DataFrame,
        expected_training_duration: int = 15 * 60,
        expected_duration: int = 60 * 60
) -> pd.DataFrame:
    """
    Removes aborted sessions

    Parameters
    ----------
    lims_connection
    behavior_df
    expected_training_duration: Expected duration for a TRAINING_0* session
        in seconds
    expected_duration: Expected duration for all non-TRAINING_0* sessions
        in seconds

    Returns
    -------
    behavior_df: behavior df with aborted sessions filtered out
    """
    durations = _get_session_duration_from_behavior_session_ids(
        lims_connection=lims_connection,
        behavior_session_id_list=(
            behavior_df['behavior_session_id'].unique().tolist())
    )
    durations = behavior_df['behavior_session_id'].map(durations)

    behavior_df = behavior_df[
        (
            (behavior_df['session_type'].str.match('TRAINING_0.*')) &
            (durations > expected_training_duration)
        ) |
        (
            ~(behavior_df['session_type'].str.match('TRAINING_0.*')) &
            (durations > expected_duration)
        )
    ]
    return behavior_df


def _get_session_duration_from_behavior_session_ids(
        lims_connection: PostgresQueryMixin,
        behavior_session_id_list: List[int]
) -> pd.Series:
    """
    Gets duration in seconds for each session in `behavior_session_id_list`

    Parameters
    ----------
    lims_connection
    behavior_session_id_list

    Returns
    -------
    pd.Series
        index: behavior_session_id
        values: duration in seconds
    """
    pickle_path_df = stimulus_pickle_paths_from_behavior_session_ids(
        lims_connection=lims_connection,
        behavior_session_id_list=behavior_session_id_list)
    durations = []
    for row in pickle_path_df.itertuples(index=False):
        durations.append({
            'behavior_session_id': row.behavior_session_id,
            'duration': (BehaviorStimulusFile(filepath=row.pkl_path)
                         .session_duration)
        })
    durations = pd.Series(
        [x['duration'] for x in durations],
        index=pd.Int64Index([x['behavior_session_id'] for x in durations],
                            name='behavior_session_id'))
    return durations
