import pytest

from allensdk.brain_observatory.behavior.behavior_project_lims_api import (
    BehaviorProjectLimsApi)
from allensdk.test_utilities.custom_comparators import (
    WhitespaceStrippedString)


class MockQueryEngine:
    def __init__(self, **kwargs):
        pass

    def select(self, query):
        return query

    def fetchall(self, query):
        return query

    def stream(self, endpoint):
        return endpoint


@pytest.fixture
def MockBehaviorProjectLimsApi():
    return BehaviorProjectLimsApi(MockQueryEngine(), MockQueryEngine())


@pytest.mark.parametrize(
    "col,valid_list,operator,expected", [
        ("os.id", [1, 2, 3], "WHERE", "WHERE os.id IN (1,2,3)"),
        ("id2", ["'a'", "'b'"], "AND", "AND id2 IN ('a','b')"),
        ("id3", [1.0], "OR", "OR id3 IN (1.0)"),
        ("id4", None, "WHERE", "")]
)
def test_build_in_list_selector_query(
        col, valid_list, operator, expected, MockBehaviorProjectLimsApi):
    assert (expected
            == MockBehaviorProjectLimsApi._build_in_list_selector_query(
                col, valid_list, operator))


@pytest.mark.parametrize(
    "behavior_session_ids,expected", [
        (None,
            WhitespaceStrippedString("""
            SELECT foraging_id
            FROM behavior_sessions
            WHERE foraging_id IS NOT NULL
            ;
            """)),
        (["'id1'", "'id2'"],
            WhitespaceStrippedString("""
            SELECT foraging_id
            FROM behavior_sessions
            WHERE foraging_id IS NOT NULL
            AND id IN ('id1','id2');
            """))
    ]
)
def test_get_foraging_ids_from_behavior_session(
        behavior_session_ids, expected, MockBehaviorProjectLimsApi):
    mock_api = MockBehaviorProjectLimsApi
    assert expected == mock_api._get_foraging_ids_from_behavior_session(
        behavior_session_ids)


def test_get_behavior_stage_table(MockBehaviorProjectLimsApi):
    expected = WhitespaceStrippedString("""
            SELECT
                stages.name as session_type,
                bs.id AS foraging_id
            FROM behavior_sessions bs
            JOIN stages ON stages.id = bs.state_id
            ;
        """)
    mock_api = MockBehaviorProjectLimsApi
    actual = mock_api._get_behavior_stage_table(mtrain_db=MockQueryEngine())
    assert expected == actual


@pytest.mark.parametrize(
    "ophys_session_ids,expected", [
        (None, WhitespaceStrippedString("""
            SELECT
                os.id as ophys_session_id,
                bs.id as behavior_session_id,
                experiment_ids as ophys_experiment_id,
                os.specimen_id,
                os.isi_experiment_id,
                os.stimulus_name as session_type,
                os.date_of_acquisition,
                d.full_genotype as genotype,
                g.name as sex,
                DATE_PART('day', os.date_of_acquisition - d.date_of_birth)
                    AS age_in_days
            FROM ophys_sessions os
            JOIN behavior_sessions bs ON os.id = bs.ophys_session_id
            JOIN donors d ON d.id = bs.donor_id
            JOIN genders g ON g.id = d.gender_id
            JOIN (-- -- begin getting all ophys_experiment_ids -- --
            SELECT
                (ARRAY_AGG(DISTINCT(oe.id))) as experiment_ids, os.id
            FROM ophys_sessions os
            RIGHT JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
            GROUP BY os.id
            -- -- end getting all ophys_experiment_ids -- --
            ) exp_ids ON os.id = exp_ids.id;
        """))]
)
def test_get_session_table(ophys_session_ids, expected,
                           MockBehaviorProjectLimsApi):
    actual = MockBehaviorProjectLimsApi._get_session_table()
    assert expected == actual
