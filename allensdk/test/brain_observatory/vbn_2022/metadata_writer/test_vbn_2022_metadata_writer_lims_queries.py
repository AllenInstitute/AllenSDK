import pandas as pd
import pytest
import datetime

from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisition
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.subject_metadata import \
    SubjectMetadata
from allensdk.brain_observatory.vbn_2022.metadata_writer.lims_queries import (
    _merge_ecephys_id_and_failed)
from allensdk.internal.brain_observatory.mouse import Mouse
from allensdk.internal.brain_observatory.util.multi_session_utils import \
    remove_invalid_sessions
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class TestLimsQueries(LimsTest):
    @pytest.mark.requires_bamboo
    def test_exclude_deceased_mice(self):
        """Tests whether deceased mice are excluded"""

        # This session is known to have a behavior session that falls after
        # death date
        ecephys_session_id = 1071300149

        behavior_session_id = BehaviorSessionId.from_ecephys_session_id(
            db=self.dbconn,
            ecephys_session_id=ecephys_session_id
        )
        mouse = Mouse.from_behavior_session_id(
            behavior_session_id=behavior_session_id.value)
        obtained = mouse.get_behavior_sessions(exclude_invalid_sessions=True)
        obtained_all = mouse.get_behavior_sessions(
            exclude_invalid_sessions=False)

        assert all([x.date_of_acquisition <=
                    x.subject_metadata.get_death_date() for x in obtained])
        assert len([x.subject_metadata.get_death_date() >
                    x.date_of_acquisition for x in obtained_all]) >= 1


def test_filter_on_death_date():
    """
    Test that remove_invalid_sessions drops the correct sessions_
    """
    sessions = []
    for is_valid in (True, False):
        sessions += [
            BehaviorMetadata(
                date_of_acquisition=DateOfAcquisition(
                    date_of_acquisition=datetime.datetime(2020, 6, 7)
                ),
                behavior_session_id=i,
                behavior_session_uuid=None,
                equipment=None,
                session_type=None,
                stimulus_frame_rate=None,
                subject_metadata=SubjectMetadata(
                    age=None,
                    driver_line=None,
                    full_genotype=None,
                    mouse_id=i,
                    reporter_line=None,
                    sex=None,
                    death_on=(
                        datetime.datetime(2020, 6, 8) if is_valid else
                        datetime.datetime(2020, 6, 6)
                    )
                )
            ) for i in range(2)]

    actual = remove_invalid_sessions(
        behavior_sessions=sessions,
        remove_pretest_sessions=False,
        remove_aborted_sessions=False,
        remove_sessions_after_mouse_death_date=True
    )
    expected = [
        sessions[0],
        sessions[1]
    ]

    assert actual == expected


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
