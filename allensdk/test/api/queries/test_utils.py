import pytest

from allensdk.internal.api.queries.utils import \
    _convert_list_of_string_to_sql_safe_string


@pytest.mark.parametrize(
    'strings, expected',
    (
        (['A', 'B'], ["'A'", "'B'"]),
        (["'A'", "'B'"], ["'A'", "'B'"]),
        (['A'], ["'A'"]),
        ([], [])
    )
)
def test_convert_list_of_string_to_sql_safe_string(strings, expected):
    assert _convert_list_of_string_to_sql_safe_string(strings=strings) == \
           expected
