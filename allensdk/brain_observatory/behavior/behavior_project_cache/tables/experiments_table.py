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
from allensdk.brain_observatory.behavior.metadata.behavior_ophys_metadata \
    import BehaviorOphysMetadata


class ExperimentsTable(ProjectTable, OphysMixin):
    def __init__(self, df: pd.DataFrame,
                 sessions_table: SessionsTable,
                 suppress: Optional[List[str]] = None):
        self._sessions_table = sessions_table

        super().__init__(df=df, suppress=suppress, experiment_level=True)

    def postprocess_additional(self):
        self._df['indicator'] = self._df['reporter_line'].apply(
            BehaviorOphysMetadata.parse_indicator)

        self._df = self._add_prior_exposures(
            sessions_table=self._sessions_table, df=self._df)
