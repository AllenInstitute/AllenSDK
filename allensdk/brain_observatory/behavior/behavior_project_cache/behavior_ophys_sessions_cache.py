import logging
import re
from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.cache_table import \
    CacheTable


class BehaviorOphysSessionsCacheTable(CacheTable):
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None,
                 by: str = 'ophys_session_id'):

        self._logger = logging.getLogger(self.__class__.__name__)
        self._by = by
        super().__init__(df=df, suppress=suppress)

    def postprocess_additional(self):
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

        # Possibly explode and reindex
        self.__explode()

    def __explode(self):
        if self._by == "ophys_session_id":
            pass
        elif self._by == "ophys_experiment_id":
            self._df = (self._df.reset_index()
                        .explode("ophys_experiment_id")
                        .set_index("ophys_experiment_id"))
        else:
            self._logger.warning(
                f"Invalid value for `by`, '{self._by}', passed to "
                f"BehaviorOphysSessionsCacheTable."
                " Valid choices for `by` are 'ophys_experiment_id' and "
                "'ophys_session_id'.")
