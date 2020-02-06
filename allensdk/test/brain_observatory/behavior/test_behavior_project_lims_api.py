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
    return BehaviorProjectLimsApi(MockQueryEngine(), MockQueryEngine(), 
                                  MockQueryEngine())


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
    actual = mock_api._get_behavior_stage_table()
    assert expected == actual


@pytest.mark.parametrize(
    "line,expected", [
        ("reporter", WhitespaceStrippedString(
            """-- -- begin getting reporter line from donors -- --
            SELECT ARRAY_AGG (g.name) AS reporter_line, d.id AS donor_id
            FROM donors d
            LEFT JOIN donors_genotypes dg ON dg.donor_id=d.id
            LEFT JOIN genotypes g ON g.id=dg.genotype_id
            LEFT JOIN genotype_types gt ON gt.id=g.genotype_type_id
            WHERE gt.name='reporter'
            GROUP BY d.id
            -- -- end getting reporter line from donors -- --""")),
        ("driver", WhitespaceStrippedString(
            """-- -- begin getting driver line from donors -- --
            SELECT ARRAY_AGG (g.name) AS driver_line, d.id AS donor_id
            FROM donors d
            LEFT JOIN donors_genotypes dg ON dg.donor_id=d.id
            LEFT JOIN genotypes g ON g.id=dg.genotype_id
            LEFT JOIN genotype_types gt ON gt.id=g.genotype_type_id
            WHERE gt.name='driver'
            GROUP BY d.id
            -- -- end getting driver line from donors -- --"""))
    ]
)
def test_build_line_from_donor_query(line, expected, 
                                     MockBehaviorProjectLimsApi):
    mbp_api = MockBehaviorProjectLimsApi
    assert expected == mbp_api._build_line_from_donor_query(line=line)
