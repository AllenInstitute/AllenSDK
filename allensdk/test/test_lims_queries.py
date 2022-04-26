import pytest

from allensdk.internal.api.queries.lims_queries import (
    build_in_list_selector_query,
    foraging_id_map_from_behavior_session_id,
    _sanitize_uuid_list)

from allensdk.internal.api.queries.mtrain_queries import (
    session_stage_from_foraging_id)

from allensdk.test_utilities.custom_comparators import (
    WhitespaceStrippedString)


@pytest.mark.parametrize(
    "col,valid_list,operator,expected", [
        ("os.id", [1, 2, 3], "WHERE", "WHERE os.id IN (1,2,3)"),
        ("id2", ["'a'", "'b'"], "AND", "AND id2 IN ('a','b')"),
        ("id3", [1.0], "OR", "OR id3 IN (1.0)"),
        ("id4", None, "WHERE", "")]
)
def test_build_in_list_selector_query(
        col, valid_list, operator, expected):
    assert (expected
            == build_in_list_selector_query(
                col, valid_list, operator))


class MockQueryEngine:
    def __init__(self, **kwargs):
        pass

    def select(self, query):
        return query

    def fetchall(self, query):
        return query

    def stream(self, endpoint):
        return endpoint

    def query(self, _query):
        return _query


@pytest.mark.parametrize(
    "behavior_session_ids,expected", [
        (None,
            WhitespaceStrippedString("""
            SELECT foraging_id, id as behavior_session_id
            FROM behavior_sessions
            WHERE foraging_id IS NOT NULL
            ;
            """)),
        (["'id1'", "'id2'"],
            WhitespaceStrippedString("""
            SELECT foraging_id, id as behavior_session_id
            FROM behavior_sessions
            WHERE foraging_id IS NOT NULL
            AND id IN ('id1','id2');
            """))
    ]
)
def test_foraging_id_map(
        behavior_session_ids, expected):
    assert expected == foraging_id_map_from_behavior_session_id(
                           lims_engine=MockQueryEngine(),
                           logger=None,
                           behavior_session_ids=behavior_session_ids)


def test_session_stage():
    expected = WhitespaceStrippedString("""
            SELECT
                stages.name as session_type,
                bs.id AS foraging_id
            FROM behavior_sessions bs
            JOIN stages ON stages.id = bs.state_id
            ;
        """)
    actual = session_stage_from_foraging_id(
                mtrain_engine=MockQueryEngine(),
                foraging_ids=None,
                logger=None)
    assert expected == actual


def test_sanitize_uuid_list():
    """
    Test that _sanitize_uuid_list gives the expected result
    """

    input_list = [
        '12345678123456781234567812345678',
        'aaa',
        '1234567812345678123456781234567812345678',
        'abcdefab-1234-abcd-0123-0123456789ab']

    sanitized = _sanitize_uuid_list(
                    uuid_list=input_list)

    assert sanitized == ['12345678123456781234567812345678',
                         'abcdefab-1234-abcd-0123-0123456789ab']
