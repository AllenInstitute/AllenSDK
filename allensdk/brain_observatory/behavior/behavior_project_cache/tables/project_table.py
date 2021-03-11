import re
from abc import abstractmethod
from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata


class ProjectTable:
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None,
                 all_sessions=False,
                 experiment_level=False):
        self._df = df
        self._suppress = suppress
        self._all_sessions = all_sessions
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

        self.__add_session_number()

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
