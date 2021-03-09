from abc import abstractmethod
from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata


class CacheTable:
    def __init__(self, df: pd.DataFrame,
                 suppress: Optional[List[str]] = None):
        self._df = df
        self._suppress = suppress

        self.postprocess()

    @property
    def table(self):
        return self._df

    def postprocess_base(self):
        self._df['reporter_line'] = self._df['reporter_line'].apply(
            BehaviorMetadata.parse_reporter_line)
        self._df['cre_line'] = self._df['full_genotype'].apply(
            BehaviorMetadata.parse_cre_line)

    def postprocess(self):
        self.postprocess_base()
        self.postprocess_additional()

        if self._suppress:
            self._df.drop(columns=self._suppress, inplace=True,
                          errors="ignore")

    @abstractmethod
    def postprocess_additional(self):
        raise NotImplemented()
