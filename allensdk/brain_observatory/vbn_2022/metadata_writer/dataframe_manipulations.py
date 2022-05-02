# This module contains utility methods for extending
# the VBN 2022 metadata dataframes as they are directly queried
# from LIMS.

import pandas as pd
import numpy as np
import pathlib
import json
import warnings

from allensdk.internal.api import PostgresQueryMixin

from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .stimulus_timestamps.timestamps_processing import (
        get_frame_indices)

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile)

from allensdk.internal.api.queries.lims_queries import (
    stimulus_pickle_paths_from_behavior_session_ids)

from allensdk.brain_observatory.vbn_2022 \
    .metadata_writer.id_generator import (
        FileIDGenerator)


def add_file_paths_to_session_table(
        session_table: pd.DataFrame,
        id_generator: FileIDGenerator,
        file_dir: pathlib.Path,
        file_prefix: str,
        index_col: str,
        on_missing_file: str) -> pd.DataFrame:
    """
    Add file_id and file_path columns to session dataframe.

    Parameters
    ----------
    session_table: pd.DataFrame
        The dataframe to which we are adding
        file_id and file_path

    id_generator: FileIDGenerator
        For maintaining a unique mapping between file_path and file_id

    file_dir: pathlib.Path
        directory where files will be found

    file_prefix: str
        Prefix of file names

    index_col: str
        Column in session_table used to index files

    on_missing_file: str
        Specifies how to handle missing files
            'error' -> raise an exception
            'warning' -> assign dummy file_id and warn
            'skip' -> drop that row from the table and warn

    Returns
    -------
    session_table:
        The same as the input dataframe but with file_id and file_path
        columns added

    Notes
    -----
    Files are assumed to be named like
    {file_dir}/{file_prefix}_{session_table.index_col}.nwb
    """

    if on_missing_file not in ('error', 'warn', 'skip'):
        msg = ("on_missing_file must be one of ('error', "
               "'warn', or 'skip'); you passed in "
               f"{on_missing_file}")
        raise ValueError(msg)

    file_suffix = 'nwb'
    new_data = []
    missing_files = []
    for file_index in session_table[index_col].values:
        file_path = file_dir / f'{file_prefix}_{file_index}.{file_suffix}'
        if not file_path.exists():
            file_id = id_generator.dummy_value
            missing_files.append(file_path.resolve().absolute())
        else:
            file_id = id_generator.id_from_path(file_path=file_path)
        str_path = str(file_path.resolve().absolute())
        new_data.append(
            {'file_id': file_id,
             'file_path': str_path,
             index_col: file_index})

    if len(missing_files) > 0:
        msg = "The following files do not exist:"
        for file_path in missing_files:
            msg += f"\n{file_path}"
        if on_missing_file == 'error':
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

    new_df = pd.DataFrame(data=new_data)
    session_table = session_table.join(
                new_df.set_index(index_col),
                on=index_col,
                how='left')

    if on_missing_file == 'skip' and len(missing_files) > 0:
        session_table = session_table.drop(
            session_table.loc[
                session_table.file_id == id_generator.dummy_value].index)

    return session_table


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


def _add_prior_omissions_to_ecephys(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the 'prior_exposures_to_omissions' columns to a dataframe
    of ecephys sessions. Return that dataframe with the column
    added.
    """

    # From communication with Corbett Bennett:
    # As for omissions, the only scripts that have them are
    # the EPHYS scripts. So prior exposure to omissions is
    # just a matter of labeling whether this was the first EPHYS
    # day or the second.
    #
    # which I take to mean that prior_exposure_to_omissions should
    # just be session_number-1 (so it is 0 on the first day, 1 on
    # the second day, etc.)

    sessions_df['prior_exposures_to_omissions'] = (
        sessions_df['session_number'] - 1)
    return sessions_df


def _add_prior_omissions_to_behavior(
        behavior_df: pd.DataFrame,
        ecephys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'prior_exposures_to_omissions' column to behavior sessions
    dataframe. Return that dataframe with the column added.

    Parameters
    ----------
    behavior_df: pd.DataFrame
        A dataframe of behavior sessions. This is the dataframe
        to which `prior_exposures_to_omissions` will be added

    ecephys_df: pd.DataFrame
        A dataframe of ecephys_sessions which contains
        `prior_exposures_to_omissions`. This will be used to add
        `prior_exposures_to_omissions` as described in the notes
        below.

    Returns
    -------
    behavior_df: pd.DataFrame
        The input behavior sessions dataframe with
        the 'prior_exposures_to_omissions` column added.

    Notes
    -----
    Because, in this project, only the ecephys sessions include
    omissions, this function sets prior_exposures_to_omissions
    to be equal to the number of ecephys_sessions the mouse
    has already seen.

    In cases where the date of the behavior session exactly
    matches the date of the ecephys session, the mouse is assumed
    not to have seen the ecephys session (i.e.
    `prior_exposures_to_omissions` will assume the ecephys
    session has not happened yet)
    """
    mouse_col = 'mouse_id'

    mouse_id_values = np.unique(behavior_df[mouse_col].values)
    new_data = []
    for mouse_id in mouse_id_values:
        sub_beh = behavior_df.query(f"{mouse_col}=='{mouse_id}'")
        sub_ecephys = ecephys_df.query(f"{mouse_col}=='{mouse_id}'")

        ecephys_dates = []
        ecephys_prior = []
        for date, prior in zip(sub_ecephys.date_of_acquisition,
                               sub_ecephys.prior_exposures_to_omissions):
            ecephys_dates.append(date.to_julian_date())
            ecephys_prior.append(prior)
        ecephys_dates = np.sort(np.array(ecephys_dates))

        beh_id_arr = []
        beh_dates = []
        for beh_id, date in zip(sub_beh.behavior_session_id,
                                sub_beh.date_of_acquisition):
            beh_dates.append(date.to_julian_date())
            beh_id_arr.append(beh_id)

        insert_indices = get_frame_indices(
                            frame_timestamps=ecephys_dates,
                            event_timestamps=beh_dates)

        for beh_id, beh_date, idx in zip(beh_id_arr,
                                         beh_dates,
                                         insert_indices):

            if np.allclose(beh_date,
                           ecephys_dates[idx],
                           rtol=0.0,
                           atol=1.0e-6):
                n_omissions = idx
            elif beh_date < ecephys_dates[idx]:
                n_omissions = max(0, idx-1)
            else:
                n_omissions = idx+1

            new_data.append(
                {'behavior_session_id': beh_id,
                 'prior_exposures_to_omissions': n_omissions})

    new_data = pd.DataFrame(data=new_data)
    behavior_df = behavior_df.join(
            new_data.set_index('behavior_session_id'),
            on='behavior_session_id',
            how='left')
    return behavior_df


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
                            behavior_session_id_list=invalid_beh)

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
