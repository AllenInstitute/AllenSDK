import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .sessions_table import \
    SessionsTable


class OphysMixin:
    @staticmethod
    def _add_prior_exposures(sessions_table: SessionsTable, df: pd.DataFrame):
        prior_exposures = sessions_table.get_prior_exposures()
        df = df.merge(prior_exposures,
                      left_on='behavior_session_id',
                      right_index=True)
        return df
