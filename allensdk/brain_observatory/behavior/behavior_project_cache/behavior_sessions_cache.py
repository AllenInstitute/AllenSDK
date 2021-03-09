from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.cache_table import \
    CacheTable


class BehaviorSessionsCacheTable(CacheTable):
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None):
        super().__init__(df=df, suppress=suppress)

    def postprocess_additional(self):
        pass


