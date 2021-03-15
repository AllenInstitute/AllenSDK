from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .ophys_mixin import OphysMixin
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .project_table import \
    ProjectTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .sessions_table import \
    SessionsTable


class ExperimentsTable(ProjectTable, OphysMixin):
    """Class for storing and manipulating project-level data
    at the behavior-ophys experiment level"""
    def __init__(self, df: pd.DataFrame,
                 sessions_table: SessionsTable,
                 suppress: Optional[List[str]] = None):
        """
        Parameters
        ----------
        df
            The behavior-ophys experiment-level data
        sessions_table
            All session-level data (needed to calculate exposure counts)
        suppress
            columns to drop from table
        """

        self._sessions_table = sessions_table

        ProjectTable.__init__(self, df=df, suppress=suppress)

    def postprocess_additional(self):
        self._df = self._add_prior_exposures(
            sessions_table=self._sessions_table, df=self._df)
