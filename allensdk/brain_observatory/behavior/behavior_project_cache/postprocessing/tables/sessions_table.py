from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache\
    .postprocessing.tables.project_table import \
    ProjectTable


class SessionsTable(ProjectTable):
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None):
        super().__init__(df=df, suppress=suppress, behavior_only=True)

    def postprocess_additional(self):
        pass


