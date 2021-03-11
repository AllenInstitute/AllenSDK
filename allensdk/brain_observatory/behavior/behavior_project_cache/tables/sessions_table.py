from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .project_table import \
    ProjectTable


class SessionsTable(ProjectTable):
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None):
        super().__init__(df=df, suppress=suppress, all_sessions=True)

    def postprocess_additional(self):
        self._df['prior_exposures_to_session_type'] = \
            self.__get_prior_exposures_to_session_type()
        self._df['prior_exposures_to_image_set'] = \
            self.__get_prior_exposures_to_image_set()

    @property
    def prior_exposures(self) -> pd.DataFrame:
        """Returns all the prior exposure values,
        with index of behavior_session_id"""
        return self._df[['prior_exposures_to_session_type',
                         'prior_exposures_to_image_set']]

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
