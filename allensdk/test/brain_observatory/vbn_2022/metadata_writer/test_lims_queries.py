import pandas as pd
import datetime

from allensdk.brain_observatory.vbn_2022.metadata_writer.lims_queries import (
    _filter_on_death_date)


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
         'death_on': datetime.datetime(2020, 7, 14)}]

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
         'date_of_acquisition': datetime.datetime(2020, 4, 7)}]

    session_df = pd.DataFrame(data=session_data)

    actual = _filter_on_death_date(
                behavior_session_df=session_df,
                lims_connection=DummyConnection())

    expected = pd.DataFrame(
            data=[session_data[1],
                  session_data[2],
                  session_data[4],
                  session_data[5]])

    expected = expected.set_index("session_id")
    actual = actual.set_index("session_id")
    pd.testing.assert_frame_equal(expected, actual)
