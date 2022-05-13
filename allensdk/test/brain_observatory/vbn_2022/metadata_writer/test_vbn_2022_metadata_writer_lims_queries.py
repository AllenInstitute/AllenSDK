import pandas as pd
import pytest

from allensdk.api.queries.donors_queries import get_death_date_for_mouse_ids
from allensdk.brain_observatory.vbn_2022.metadata_writer.lims_queries import \
    _behavior_session_table_from_ecephys_session_id_list
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestLimsQueries(LimsTest):
    @pytest.mark.requires_bamboo
    def test_exclude_deceased_mice(self):
        """Tests whether deceased mice are excluded"""

        # This session is known to have a behavior session that falls after
        # death date
        ecephys_session_id = 1071300149

        obtained = \
            _behavior_session_table_from_ecephys_session_id_list(
                lims_connection=self.dbconn,
                mtrain_connection=self.mtrainconn,
                ecephys_session_id_list=[ecephys_session_id]
            )
        obtained_include = \
            _behavior_session_table_from_ecephys_session_id_list(
                lims_connection=self.dbconn,
                mtrain_connection=self.mtrainconn,
                ecephys_session_id_list=[ecephys_session_id],
                exclude_sessions_after_death_date=False
            )
        death_date = get_death_date_for_mouse_ids(
            lims_connections=self.dbconn,
            mouse_ids_list=obtained['mouse_id'].unique().tolist())
        obtained['death_date'] = death_date.loc[0, 'death_on']
        obtained['death_date'] = pd.to_datetime(obtained['death_date'])
        obtained['date_of_acquisition'] = pd.to_datetime(
            obtained['date_of_acquisition'])

        assert (obtained['date_of_acquisition'] <=
                obtained['death_date']).all()
        assert obtained.shape[0] == obtained_include.shape[0] - 1
