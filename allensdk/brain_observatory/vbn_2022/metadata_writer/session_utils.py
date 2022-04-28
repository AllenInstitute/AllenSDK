from typing import List
import pandas as pd
import numpy as np
import json

from allensdk.internal.api import PostgresQueryMixin

from allensdk.brain_observatory.behavior.data_objects.timestamps \
    .stimulus_timestamps.timestamps_processing import (
        get_frame_indices)

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile)

from allensdk.internal.api.queries.lims_queries import (
    behavior_sessions_from_ecephys_session_ids,
    foraging_id_map_from_behavior_session_id,
    stimulus_pickle_paths_from_behavior_session_ids,
    _sanitize_uuid_list)

from allensdk.internal.api.queries.mtrain_queries import (
    session_stage_from_foraging_id)

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .util.prior_exposure_processing import (
        get_image_set,
        get_prior_exposures_to_image_set,
        get_prior_exposures_to_session_type)


def _add_session_number(
        sessions_df: pd.DataFrame,
        index_col: str) -> pd.DataFrame:
    """
    Parses session number from session type and and adds to dataframe

    index_col should be either "ecephys_session_id" or "behavior_session_id"
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


def _add_prior_images(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    sessions_df['prior_exposures_to_image_set'] = \
            get_prior_exposures_to_image_set(df=sessions_df)
    return sessions_df


def _add_prior_omissions_ecephys(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    # From communication with Corbett Bennett:
    # As for omissions, the only scripts that have them are
    # the EPHYS scripts. So prior exposure to omissions is
    # just a matter of labeling whether this was the first EPHYS
    # day or the second.
    #
    # which I take to mean that prior_exposure_to_omissions should
    # just be session_number-1 (so it is 0 on the first day, 1 on
    # the second day, etc.)

    sessions_df['prior_exposures_to_omissions'] = \
                sessions_df['session_number'] - 1
    return sessions_df


def _add_prior_omissions_behavior(
        behavior_df: pd.DataFrame,
        ecephys_df: pd.DataFrame) -> pd.DataFrame:
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
        ecephys_dates = np.array(ecephys_dates)
        ecephys_prior = np.array(ecephys_prior)
        sorted_dex = np.argsort(ecephys_dates)
        ecephys_dates = ecephys_dates[sorted_dex]
        ecephys_prior = ecephys_prior[sorted_dex]

        beh_id_arr = []
        beh_dates = []
        for beh_id, date in zip(sub_beh.behavior_session_id,
                                sub_beh.date_of_acquisition):
            beh_dates.append(date.to_julian_date())
            beh_id_arr.append(beh_id)

        insert_indices = get_frame_indices(
                            frame_timestamps=ecephys_dates,
                            event_timestamps=beh_dates)

        for beh_id, idx in zip(beh_id_arr, insert_indices):
            new_data.append(
                {'behavior_session_id': beh_id,
                 'prior_exposures_to_omissions': ecephys_prior[idx]})

    new_data = pd.DataFrame(data=new_data)
    behavior_df = behavior_df.join(
            new_data.set_index('behavior_session_id'),
            on='behavior_session_id',
            how='left')
    return behavior_df


def _add_experience_level(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    sessions_df['experience_level'] = np.where(
                  np.logical_or(
                      sessions_df['prior_exposures_to_image_set'] == 0,
                      sessions_df['prior_exposures_to_image_set'].isnull()),
                  'Novel',
                  'Familiar')
    return sessions_df


def _patch_df_from_pickle_file(
        lims_connection: PostgresQueryMixin,
        behavior_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a DataFrame and patch date_of_acquisition, session_type
    from the pickle file
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
    # get age in days
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


def behavior_session_table_from_ecephys_session_id(
        lims_connection: PostgresQueryMixin,
        mtrain_connection: PostgresQueryMixin,
        ecephys_session_ids: List[int]) -> pd.DataFrame:
    """
    """
    behavior_session_df = behavior_sessions_from_ecephys_session_ids(
                            lims_connection=lims_connection,
                            ecephys_session_id_list=ecephys_session_ids)

    beh_to_foraging_df = foraging_id_map_from_behavior_session_id(
        lims_engine=lims_connection,
        behavior_session_ids=behavior_session_df.behavior_session_id.tolist())

    # add foraging_id
    behavior_session_df = behavior_session_df.join(
                    beh_to_foraging_df.set_index('behavior_session_id'),
                    on='behavior_session_id',
                    how='left')

    foraging_ids = beh_to_foraging_df.foraging_id.tolist()

    # this is necessary because there are some sessions with the
    # invalid foraging_id entry 'DoC' in MTRAIN
    foraging_ids = _sanitize_uuid_list(foraging_ids)

    foraging_to_stage_df = session_stage_from_foraging_id(
            mtrain_engine=mtrain_connection,
            foraging_ids=foraging_ids)

    # add session_type
    behavior_session_df = behavior_session_df.join(
                foraging_to_stage_df.set_index('foraging_id'),
                on='foraging_id',
                how='left')

    behavior_session_df = _patch_df_from_pickle_file(
                             lims_connection=lims_connection,
                             behavior_df=behavior_session_df)

    behavior_session_df['image_set'] = get_image_set(
            df=behavior_session_df)

    behavior_session_df['prior_exposures_to_session_type'] = \
        get_prior_exposures_to_session_type(
            df=behavior_session_df)

    behavior_session_df['prior_exposures_to_image_set'] = \
        get_prior_exposures_to_image_set(
            df=behavior_session_df)

    behavior_session_df = _add_age_in_days(
        df=behavior_session_df)

    behavior_session_df = _add_session_number(
        sessions_df=behavior_session_df,
        index_col="behavior_session_id")

    return behavior_session_df


def _postprocess_sessions(
        sessions_df: pd.DataFrame) -> pd.DataFrame:

    sessions_df = _add_session_number(sessions_df=sessions_df,
                                      index_col="ecephys_session_id")
    sessions_df = _add_prior_omissions_ecephys(sessions_df=sessions_df)
    sessions_df = _add_experience_level(sessions_df=sessions_df)

    return sessions_df
