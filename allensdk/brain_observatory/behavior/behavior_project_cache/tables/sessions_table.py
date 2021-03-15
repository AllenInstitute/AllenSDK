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
    """Class for storing and manipulating project-level data
    at the session level"""
    def __init__(self, df: pd.DataFrame,
                 fetch_api: BehaviorProjectLimsApi,
                 suppress: Optional[List[str]] = None):
        """
        Parameters
        ----------
        df
            The session-level data
        fetch_api
            The api needed to call mtrain db
        suppress
            columns to drop from table
        """
        self._fetch_api = fetch_api
        super().__init__(df=df, suppress=suppress)

    def postprocess_additional(self):
        # Note: these prior exposure counts are only calculated here
        # and then are merged into other tables
        # This is because this is the only table that contains all sessions
        # (behavior and ophys)
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
