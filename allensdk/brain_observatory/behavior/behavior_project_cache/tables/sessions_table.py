from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .project_table import \
    ProjectTable
from allensdk.brain_observatory.behavior.project_apis.data_io import \
    BehaviorProjectLimsApi


class SessionsTable(ProjectTable):
    def __init__(self, df: pd.DataFrame,
                 fetch_api: BehaviorProjectLimsApi,
                 suppress: Optional[List[str]] = None):
        self._fetch_api = fetch_api
        super().__init__(df=df, suppress=suppress, all_sessions=True)

    def postprocess_additional(self):
        self._df['prior_exposures_to_session_type'] = \
            self.__get_prior_exposures_to_session_type()
        self._df['prior_exposures_to_image_set'] = \
            self.__get_prior_exposures_to_image_set()
        self._df['prior_exposures_to_omissions'] = \
            self.__get_prior_exposures_to_omissions()

    @property
    def prior_exposures(self) -> pd.DataFrame:
        """Returns all the prior exposure values,
        with index of behavior_session_id"""
        return self._df[
            [c for c in self._df if c.startswith('prior_exposures_to')]
        ]

    def __get_prior_exposure_count(self, to: pd.Series) -> pd.Series:
        """Returns prior exposures a subject had to something
        i.e can be prior exposures to a stimulus type, a image_set or
        omission

        Parameters
        ----------
        to
            The array to calculate prior exposures to
            Needs to have the same index as self._df

        Returns
        ---------
        Series with index same as self._df and with values of prior
        exposure counts
        """
        df = self._df.sort_values('date_of_acquisition')
        df = df[df['session_type'].notnull()]

        # reindex "to" to df
        to = to.loc[df.index]

        # exclude missing values from cumcount
        to = to[to.notnull()]

        # reindex df to match "to" index with missing values removed
        df = df.loc[to.index]

        return df.groupby(['mouse_id', to]).cumcount()

    def __get_prior_exposures_to_session_type(self):
        """Get prior exposures to session type"""
        return self.__get_prior_exposure_count(to=self._df['session_type'])

    def __get_prior_exposures_to_image_set(self):
        """Get prior exposures to image set

        The image set here is the letter part of the session type
        ie for session type OPHYS_1_images_B, it would be "B"

        Some session types don't have an image set name, such as
        gratings, which will be set to null
        """
        def __get_image_set_name(session_type: Optional[str]):
            if 'images' not in session_type:
                return None

            try:
                image_set = session_type.split('_')[3]
            except IndexError:
                image_set = None
            return image_set
        session_type = self._df['session_type'][
            self._df['session_type'].notnull()]
        image_set = session_type.apply(__get_image_set_name)
        return self.__get_prior_exposure_count(to=image_set)

    def __get_prior_exposures_to_omissions(self):
        df = self._df[self._df['session_type'].notnull()]

        contains_omissions = pd.Series(False, index=df.index)

        def __get_habituation_sessions(df: pd.DataFrame):
            """Returns all habituation sessions"""
            return df[
                df['session_type'].str.lower().str.contains('habituation')]

        def __get_habituation_sessions_contain_omissions(
                habituation_sessions: pd.DataFrame) -> pd.Series:
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
            mtrain_stage_parameters = self._fetch_api.\
                get_behavior_stage_parameters(foraging_ids=foraging_ids)
            return habituation_sessions.apply(
                lambda session: __session_contains_omissions(
                    mtrain_stage_parameters=mtrain_stage_parameters.loc[
                        session['foraging_id']]), axis=1)

        habituation_sessions = __get_habituation_sessions(df=df)
        if not habituation_sessions.empty:
            contains_omissions.loc[habituation_sessions.index] = \
                __get_habituation_sessions_contain_omissions(
                    habituation_sessions=habituation_sessions)

        contains_omissions.loc[
            (df['session_type'].str.lower().str.contains('ophys')) &
            (~df.index.isin(habituation_sessions.index))
        ] = True
        contains_omissions = contains_omissions[contains_omissions]
        return self.__get_prior_exposure_count(to=contains_omissions)
