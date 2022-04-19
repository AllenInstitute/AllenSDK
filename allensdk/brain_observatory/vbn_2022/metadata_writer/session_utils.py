import pandas as pd
import numpy as np
import json

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .util.prior_exposure_processing import (
        get_image_set,
        get_prior_exposures_to_image_set)


def _add_session_number(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Parses session number from session type and and adds to dataframe"""

    index_col = 'ecephys_session_id'
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


def _add_image_set(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    image_set = get_image_set(df=sessions_df)
    sessions_df['image_set'] = image_set
    return sessions_df


def _add_prior_images(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    sessions_df['prior_exposures_to_image_set'] = \
            get_prior_exposures_to_image_set(df=sessions_df)
    return sessions_df


def _add_prior_omissions(
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


def _add_experience_level(
        sessions_df: pd.DataFrame) -> pd.DataFrame:
    sessions_df['experience_level'] = np.where(
                  np.logical_or(
                      sessions_df['prior_exposures_to_image_set'] == 0,
                      sessions_df['prior_exposures_to_image_set'].isnull()),
                  'Novel',
                  'Familiar')
    return sessions_df


def _postprocess_sessions(
        sessions_df: pd.DataFrame) -> pd.DataFrame:

    sessions_df = _add_session_number(sessions_df=sessions_df)
    sessions_df = _add_image_set(sessions_df=sessions_df)
    sessions_df = _add_prior_images(sessions_df=sessions_df)
    sessions_df = _add_prior_omissions(sessions_df=sessions_df)
    sessions_df = _add_experience_level(sessions_df=sessions_df)

    return sessions_df
