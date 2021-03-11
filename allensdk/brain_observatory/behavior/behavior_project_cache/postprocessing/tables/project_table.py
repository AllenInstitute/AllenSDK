import re
from abc import abstractmethod
from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata


class ProjectTable:
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None,
                 behavior_only=False,
                 experiment_level=False):
        self._df = df
        self._suppress = suppress
        self._behavior_only = behavior_only
        self._experiment_level = experiment_level

        self.postprocess()

    @property
    def table(self):
        return self._df

    def postprocess_base(self):
        # Make sure the index is not duplicated (it is rare)
        self._df = self._df[~self._df.index.duplicated()]

        self._df['reporter_line'] = self._df['reporter_line'].apply(
            BehaviorMetadata.parse_reporter_line)
        self._df['cre_line'] = self._df['full_genotype'].apply(
            BehaviorMetadata.parse_cre_line)

        if not self._behavior_only:
            self.__add_session_number()

        self._df['prior_exposures_to_session_type'] = \
            self.__get_prior_exposures_to_session_type()
        self._df['prior_exposures_to_image_set'] = \
            self.__get_prior_exposures_to_image_set()

    def postprocess(self):
        self.postprocess_base()
        self.postprocess_additional()

        if self._suppress:
            self._df.drop(columns=self._suppress, inplace=True,
                          errors="ignore")

    @abstractmethod
    def postprocess_additional(self):
        raise NotImplemented()

    def __add_session_number(self):
        def parse_session_number(session_type: str):
            """Parse the session number from session type"""
            match = re.match(r'OPHYS_(?P<session_number>\d+)',
                             session_type)
            if match is None:
                return None
            return int(match.group('session_number'))

        session_type = self._df['session_type']
        session_type = session_type[session_type.notnull()]

        self._df.loc[session_type.index, 'session_number'] = \
            session_type.apply(parse_session_number)

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

        if self._experiment_level:
            df = df[~df['behavior_session_id'].duplicated()]

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
        def __get_image_set_name(session_type: str):
            if 'images' not in session_type:
                return None

            try:
                image_set = session_type.split('_')[3]
            except IndexError:
                image_set = None
            return image_set
        df = self._df[self._df['session_type'].notnull()]
        image_set = df['session_type'].apply(__get_image_set_name)
        return self.__get_prior_exposure_count(to=image_set)

