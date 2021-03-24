from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .ophys_mixin import \
    OphysMixin
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .project_table import \
    ProjectTable


class ExperimentsTable(ProjectTable, OphysMixin):
    """Class for storing and manipulating project-level data
    at the behavior-ophys experiment level"""
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None):
        """
        Parameters
        ----------
        df
            The behavior-ophys experiment-level data
        suppress
            columns to drop from table
        """

        ProjectTable.__init__(self, df=df, suppress=suppress)
        OphysMixin.__init__(self)

    def postprocess_additional(self):
        pass
