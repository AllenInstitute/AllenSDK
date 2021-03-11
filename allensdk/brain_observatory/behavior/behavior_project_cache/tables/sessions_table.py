from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .util.prior_exposure_processing import \
    get_prior_exposures_to_session_type, get_prior_exposures_to_image_set, \
    get_prior_exposures_to_omissions
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .project_table import \
    ProjectTable
from allensdk.brain_observatory.behavior.project_apis.data_io import \
    BehaviorProjectLimsApi


class SessionsTable(ProjectTable):
    def __init__(self, df: pd.DataFrame,
                 fetch_api: BehaviorProjectLimsApi,
                 suppress: Optional[List[str]] = None):
        self._fetch_api = fetch_api
        super().__init__(df=df, suppress=suppress, all_sessions=True)

    def postprocess_additional(self):
        self._df['prior_exposures_to_session_type'] = \
            get_prior_exposures_to_session_type(df=self._df)
        self._df['prior_exposures_to_image_set'] = \
            get_prior_exposures_to_image_set(df=self._df)
        self._df['prior_exposures_to_omissions'] = \
            get_prior_exposures_to_omissions(df=self._df,
                                             fetch_api=self._fetch_api)

    @property
    def prior_exposures(self) -> pd.DataFrame:
        """Returns all the prior exposure values,
        with index of behavior_session_id"""
        return self._df[
            [c for c in self._df if c.startswith('prior_exposures_to')]
        ]
