import re
from typing import Optional

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io import BehaviorProjectLimsApi  # noqa: E501


def get_prior_exposures_to_session_type(df: pd.DataFrame) -> pd.Series:
    """Get prior exposures to session type

    Parameters
    ----------
    df
        The sessions df

    Returns
    ---------
    Series with index same as df and values prior exposure counts to
    session type
    """
    return __get_prior_exposure_count(df=df, to=df['session_type'])


def get_prior_exposures_to_image_set(df: pd.DataFrame) -> pd.Series:
    """Get prior exposures to image set

    The image set here is the letter part of the session type
    ie for session type OPHYS_1_images_B, it would be "B"

    Some session types don't have an image set name, such as
    gratings, which will be set to null

    Parameters
    ----------
    df
        The session df

    Returns
    --------
    Series with index same as df and values prior exposure counts to image set
    """

    def __get_image_set_name(session_type: Optional[str]):
        match = re.match(r'.*images_(?P<image_set>\w)', session_type)
        if match is None:
            return None
        return match.group('image_set')

    session_type = df['session_type'][
        df['session_type'].notnull()]
    image_set = session_type.apply(__get_image_set_name)
    return __get_prior_exposure_count(df=df, to=image_set)


def get_prior_exposures_to_omissions(df: pd.DataFrame,
                                     fetch_api: BehaviorProjectLimsApi) -> \
        pd.Series:
    """Get prior exposures to omissions

    Parameters
    ----------
    df
        The session df
    fetch_api
        API needed to query mtrain

    Returns
    ---------
    Series with index same as df and values prior exposure counts to omissions
    """
    df = df[df['session_type'].notnull()]

    contains_omissions = pd.Series(False, index=df.index)

    def __get_habituation_sessions(df: pd.DataFrame):
        """Returns all habituation sessions"""
        return df[
            df['session_type'].str.lower().str.contains('habituation')]

    def __get_habituation_sessions_contain_omissions(
            habituation_sessions: pd.DataFrame,
            fetch_api: BehaviorProjectLimsApi) -> pd.Series:
        """Habituation sessions are not supposed to include omissions but
        because of a mistake omissions were included for some habituation
        sessions.

        This queries mtrain to figure out if omissions were included
        for any of the habituation sessions

        Parameters
        ----------
        habituation_sessions
            the habituation sessions

        Returns
        ---------
        series where index is same as habituation sessions and values
            indicate whether omissions were included
        """

        def __session_contains_omissions(
                mtrain_stage_parameters: dict) -> bool:
            return 'flash_omit_probability' in mtrain_stage_parameters \
                   and \
                   mtrain_stage_parameters['flash_omit_probability'] > 0

        foraging_ids = habituation_sessions['foraging_id'].tolist()
        foraging_ids = [f'\'{x}\'' for x in foraging_ids]
        mtrain_stage_parameters = fetch_api. \
            get_behavior_stage_parameters(foraging_ids=foraging_ids)
        return habituation_sessions.apply(
            lambda session: __session_contains_omissions(
                mtrain_stage_parameters=mtrain_stage_parameters[
                    session['foraging_id']]), axis=1)

    habituation_sessions = __get_habituation_sessions(df=df)
    if not habituation_sessions.empty:
        contains_omissions.loc[habituation_sessions.index] = \
            __get_habituation_sessions_contain_omissions(
                habituation_sessions=habituation_sessions,
                fetch_api=fetch_api)

    contains_omissions.loc[
        (df['session_type'].str.lower().str.contains('ophys')) &
        (~df.index.isin(habituation_sessions.index))
        ] = True
    return __get_prior_exposure_count(df=df, to=contains_omissions,
                                      agg_method='cumsum')


def __get_prior_exposure_count(df: pd.DataFrame, to: pd.Series,
                               agg_method='cumcount') -> pd.Series:
    """Returns prior exposures a subject had to something
    i.e can be prior exposures to a stimulus type, a image_set or
    omission

    Parameters
    ----------
    df
        The sessions df
    to
        The array to calculate prior exposures to
        Needs to have the same index as self._df
    agg_method
        The aggregation method to apply on the groups (cumcount or cumsum)

    Returns
    ---------
    Series with index same as self._df and with values of prior
    exposure counts
    """
    index = df.index
    df = df.sort_values('date_of_acquisition')
    df = df[df['session_type'].notnull()]

    # reindex "to" to df
    to = to.loc[df.index]

    # exclude missing values from cumcount
    to = to[to.notnull()]

    # reindex df to match "to" index with missing values removed
    df = df.loc[to.index]

    if agg_method == 'cumcount':
        counts = df.groupby(['mouse_id', to]).cumcount()
    elif agg_method == 'cumsum':
        df['to'] = to

        def cumsum(x):
            return x.cumsum().shift(fill_value=0).astype('int64')

        counts = df.groupby(['mouse_id'])['to'].apply(cumsum)
        counts.name = None
    else:
        raise ValueError(f'agg method {agg_method} not supported')

    # reindex to original index
    return counts.reindex(index)
