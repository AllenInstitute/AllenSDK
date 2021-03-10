import re
from abc import abstractmethod
from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata


class CacheTable:
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
            self.__get_session_type_exposure_count()
        
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

    def __get_session_type_exposure_count(self):
        df = self._df.sort_values('date_of_acquisition')
        df = df[df['session_type'].notnull()]

        if self._experiment_level:
            groupby = ['specimen_id', 'container_id', 'session_type']
        elif self._behavior_only:
            groupby = ['specimen_id', 'session_type']
        else:
            groupby = ['specimen_id', 'session_type']

        prior_exposures = df.groupby(groupby).cumcount()
        return prior_exposures
