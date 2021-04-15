import re
from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .ophys_sessions_table import \
    BehaviorOphysSessionsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .util.prior_exposure_processing import \
    get_prior_exposures_to_session_type, get_prior_exposures_to_image_set, \
    get_prior_exposures_to_omissions
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .project_table import \
    ProjectTable
from allensdk.brain_observatory.behavior.metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io import BehaviorProjectLimsApi  # noqa: E501


class SessionsTable(ProjectTable):
    """Class for storing and manipulating project-level data
    at the session level"""

    def __init__(
            self, df: pd.DataFrame,
            fetch_api: BehaviorProjectLimsApi,
            suppress: Optional[List[str]] = None,
            ophys_session_table: Optional[BehaviorOphysSessionsTable] = None):
        """
        Parameters
        ----------
        df
            The session-level data
        fetch_api
            The api needed to call mtrain db
        suppress
            columns to drop from table
        ophys_session_table
            BehaviorOphysSessionsTable, to optionally merge in ophys data
        """
        self._fetch_api = fetch_api
        self._ophys_session_table = ophys_session_table
        super().__init__(df=df, suppress=suppress)

    def postprocess_additional(self):
        self._df['reporter_line'] = self._df['reporter_line'].apply(
            BehaviorMetadata.parse_reporter_line)
        self._df['cre_line'] = self._df['full_genotype'].apply(
            BehaviorMetadata.parse_cre_line)
        self._df['indicator'] = self._df['reporter_line'].apply(
            BehaviorMetadata.parse_indicator)

        self.__add_session_number()

        self._df['prior_exposures_to_session_type'] = \
            get_prior_exposures_to_session_type(df=self._df)
        self._df['prior_exposures_to_image_set'] = \
            get_prior_exposures_to_image_set(df=self._df)
        self._df['prior_exposures_to_omissions'] = \
            get_prior_exposures_to_omissions(df=self._df,
                                             fetch_api=self._fetch_api)

        if self._ophys_session_table is not None:
            # Merge in ophys data
            self._df = self._df.reset_index() \
                .merge(self._ophys_session_table.table.reset_index(),
                       on='behavior_session_id',
                       how='left',
                       suffixes=('_behavior', '_ophys'))
            self._df = self._df.set_index('behavior_session_id')

            # Prioritize behavior date_of_acquisition
            self._df['date_of_acquisition'] = \
                self._df['date_of_acquisition_behavior']
            self._df = self._df.drop(['date_of_acquisition_behavior',
                                      'date_of_acquisition_ophys'], axis=1)

            self._df['session_type'] = \
                self.__get_session_type()
            self._df = self._df.drop(
                ['session_type_behavior',
                 'session_type_ophys'], axis=1)

    def __add_session_number(self):
        """Parses session number from session type and and adds to dataframe"""

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

    def __get_session_type(self) -> pd.Series:
        """Session type is returned by both mtrain for behavior sessions
        as well as in LIMS table ophys_sessions.

        This method applies logic to use the mtrain value for behavior-only
        sessions and LIMS value otherwise
        """
        behavior_only = self._df['ophys_session_id'].isnull()
        behavior_only_session = \
            self._df[behavior_only]['session_type_behavior']
        behavior_ophys_session = self._df[~behavior_only]['session_type_ophys']
        return pd.concat([behavior_only_session, behavior_ophys_session])
