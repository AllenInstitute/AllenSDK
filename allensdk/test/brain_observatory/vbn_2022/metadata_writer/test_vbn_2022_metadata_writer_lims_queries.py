import pandas as pd
import pytest
import datetime

from allensdk.api.queries.donors_queries import get_death_date_for_mouse_ids
from allensdk.brain_observatory.vbn_2022.metadata_writer.lims_queries import (
    _behavior_session_table_from_ecephys_session_id_list,
    _filter_on_death_date,
    _merge_ecephys_id_and_failed)
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


def test_filter_on_death_date():
    """
    Test that _filter_on_death_date drops the correct rows from
    the behavior session dataframe
    """
    death_dates = [
        {'mouse_id': 123,
         'death_on': datetime.datetime(2020, 4, 9)},
        {'mouse_id': 456,
         'death_on': datetime.datetime(2020, 3, 2)},
        {'mouse_id': 789,
         'death_on': datetime.datetime(2020, 7, 14)},
        {'mouse_id': 1011,
         'death_on': None}]

    death_df = pd.DataFrame(data=death_dates)

    class DummyConnection(object):

        def select(self, query=None):
            if "SELECT external_donor_name as mouse_id, death_on" in query:
                return death_df
            raise RuntimeError(f"not mocking query '{query}'")

    session_data = [
        {'mouse_id': 123,
         'session_id': 0,
         'date_of_acquisition': datetime.datetime(2020, 6, 7)},
        {'mouse_id': 789,
         'session_id': 1,
         'date_of_acquisition': datetime.datetime(2020, 6, 22)},
        {'mouse_id': 123,
         'session_id': 2,
         'date_of_acquisition': datetime.datetime(2020, 3, 5)},
        {'mouse_id': 456,
         'session_id': 3,
         'date_of_acquisition': datetime.datetime(2020, 4, 8)},
        {'mouse_id': 456,
         'session_id': 4,
         'date_of_acquisition': datetime.datetime(2020, 2, 2)},
        {'mouse_id': 123,
         'session_id': 5,
         'date_of_acquisition': datetime.datetime(2020, 4, 7)},
        {'mouse_id': 1011,
         'session_id': 6,
         'date_of_acqusition': datetime.datetime(2021, 11, 11)}]

    session_df = pd.DataFrame(data=session_data)

    actual = _filter_on_death_date(
                behavior_session_df=session_df,
                lims_connection=DummyConnection())

    expected = pd.DataFrame(
            data=[session_data[1],
                  session_data[2],
                  session_data[4],
                  session_data[5],
                  session_data[6]])

    expected = expected.set_index("session_id")
    actual = actual.set_index("session_id")
    pd.testing.assert_frame_equal(expected, actual)


def test_merge_ecephys_id_and_failed():
    """
    Test that method merging ecephys_session_id_list
    and failed_ecephys_session_id_list on shared donor_id
    returns the correct result
    """

    ecephys_data = [
        {'ecephys_session_id': 4,
         'donor_id': '100'},
        {'ecephys_session_id': 1,
         'donor_id': '200'},
        {'ecephys_session_id': 3,
         'donor_id': '300'},
        {'ecephys_session_id': 2,
         'donor_id': '400'}]

    failed_data = [
        {'ecephys_session_id': 7,
         'donor_id': '300'},
        {'ecphys_session_id': 6,
         'donor_id': '900'},
        {'ecephys_session_id': 5,
         'donor_id': '200'}]

    class DummyConnection(object):

        def select(self, query=None):
            if '5' in query:
                return pd.DataFrame(data=failed_data)
            elif '3' in query:
                return pd.DataFrame(data=ecephys_data)
            else:
                raise RuntimeError(
                    f"cannot mock query={query}")

    expected = [1, 2, 3, 4, 5, 7]
    actual = _merge_ecephys_id_and_failed(
                lims_connection=DummyConnection(),
                ecephys_session_id_list=[1, 2, 3, 4],
                failed_ecephys_session_id_list=[5, 6, 7])
    assert expected == actual
