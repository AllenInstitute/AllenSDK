import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .sessions_table import \
    SessionsTable


class OphysMixin:
    """A mixin for ophys data"""
    @staticmethod
    def _add_prior_exposures(sessions_table: SessionsTable,
                             df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds prior exposures by merging from sessions_table

        Parameters
        ----------
        df
            The behavior-ophys session-level data
        sessions_table
            sessions table to merge from
        """
        prior_exposures = sessions_table.prior_exposures
        df = df.merge(prior_exposures,
                      left_on='behavior_session_id',
                      right_index=True)
        return df
