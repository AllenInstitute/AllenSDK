import pytest
import pandas as pd

from allensdk.internal.api.queries.utils import (
    build_in_list_selector_query,
    _sanitize_uuid_list, build_where_clause)

from allensdk.internal.api.queries.behavior_lims_queries import (
    foraging_id_map_from_behavior_session_id)

from allensdk.internal.api.queries.ecephys_lims_queries import (
    donor_id_list_from_ecephys_session_ids)

from allensdk.test_utilities.custom_comparators import (
    WhitespaceStrippedString)


@pytest.mark.parametrize(
    "col,valid_list,operator,valid,expected", [
        ("os.id", [1, 2, 3], "WHERE", True, "WHERE os.id IN (1,2,3)"),
        ("id2", ["'a'", "'b'"], "AND", True, "AND id2 IN ('a','b')"),
        ("id3", [1.0], "OR", True, "OR id3 IN (1.0)"),
        ("id4", None, "WHERE", True, ""),
        ("os.id", [1, 2, 3], "WHERE", False, "WHERE os.id NOT IN (1,2,3)"),
        ("id2", ["'a'", "'b'"], "AND", False, "AND id2 NOT IN ('a','b')"),
        ("id3", [1.0], "OR", False, "OR id3 NOT IN (1.0)"),
        ("id4", None, "WHERE", False, "")]
)
def test_build_in_list_selector_query(
        col, valid_list, operator, valid, expected):
    assert (expected
            == build_in_list_selector_query(
                col=col,
                valid_list=valid_list,
                operator=operator,
                valid=valid))


def test_build_in_selector_error():
    """
    Test that build_in_list_selector_query raises the
    expected error for an invalid operator
    """
    with pytest.raises(ValueError, match='Operator must be'):
        build_in_list_selector_query(
            col='silly',
            valid_list=[1, 2, 3],
            operator='above',
            valid=True)


@pytest.mark.parametrize('clauses, expected', [
    (['foo=b', 'baz=a'], 'WHERE foo=b AND baz=a'),
    (['WHERE foo=b and baz=c'], 'WHERE foo=b and baz=c'),
    (['foo=b', 'baz=a', 'bar=c'], 'WHERE foo=b AND baz=a AND bar=c'),
    (['WHERE foo=b', 'baz=a'], 'WHERE foo=b AND baz=a'),
    (['where foo=b', 'baz=a'], 'where foo=b AND baz=a'),
    ([], ''),
    (['foo=b'], 'WHERE foo=b')
])
def test_build_where_clause(clauses, expected):
    assert build_where_clause(clauses=clauses) == expected


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


def test_donor_id_list_from_ecephys_session_ids():
    """
    Test that donor_id_list_from_ecephys_session_ids returns
    a sorted list of donor_ids

    (assumes that donor_lookup_from_ecephys_session_ids
    works properly)
    """

    class DummyConnection(object):

        def select(self, query=None):
            data = [{'ecephys_session_id': 1, 'donor_id': 2},
                    {'ecephys_session_id': 3, 'donor_id': 2},
                    {'ecephys_session_id': 4, 'donor_id': 0},
                    {'ecephys_session_id': 5, 'donor_id': 7},
                    {'ecephys_session_id': 6, 'donor_id': 2}]

            return pd.DataFrame(data=data)

    expected = [0, 2, 7]

    actual = donor_id_list_from_ecephys_session_ids(
            lims_connection=DummyConnection(),
            session_id_list=[9, 9, 9])

    assert actual == expected
