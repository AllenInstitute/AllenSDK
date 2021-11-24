import re
from typing import Union
import difflib
import pandas as pd
import numpy as np


class WhitespaceStrippedString(object):
    """Comparator class to compare strings that have been stripped of
    whitespace. By default removes any unicode whitespace character that
    matches the regex \\s, (which includes [ \\t\\n\\r\\f\\v],
    and other unicode whitespace characters).
    """
    def __init__(self, string: str, whitespace_chars: str = r"\s",
                 ASCII: bool = False):
        self.orig = string
        self.whitespace_chars = whitespace_chars
        self.flags = re.ASCII if ASCII else 0
        self.differ = difflib.Differ()
        self.value = re.sub(self.whitespace_chars, "", string, self.flags)

    def __eq__(self, other: Union[str, "WhitespaceStrippedString"]):
        if isinstance(other, str):
            other = WhitespaceStrippedString(
                other, self.whitespace_chars, self.flags)
        self.diff = list(self.differ.compare(self.value, other.value))
        return self.value == other.value


def safe_df_comparison(expected: pd.DataFrame,
                       obtained: pd.DataFrame,
                       expect_identical_column_order: bool = False):
    """
    Compare two dataframes in a way that is agnostic to column order
    and datatype of NULL values

    Parameters
    ----------
    expected: pd.DataFrame

    obtained: pd.DataFrame

    expect_identical_column_order: bool
       If True, raise an error if columns are not
       in the same order (default=False)

    Raises
    ------
    RuntimeError
       If:
           - dataframes do not have the same columns
           - dataframes do not have identical indexes
           - dataframe columns do not have identical contents

        When comparing the contents of dataframe columns,
        the function:
            - verifies that NULL values (whether None or NaN) are in
              the same location
            - loops over non-null values, casts arrays into lists, and
              compares with ==
    """
    msg = ''
    columns_match = True
    if not expect_identical_column_order:
        obtained_column_set = set(obtained.columns)
        expected_column_set = set(expected.columns)
        if obtained_column_set != expected_column_set:
            columns_match = False
    else:
        if not obtained.columns.equals(expected.columns):
            columns_match = False

    if not columns_match:
        msg += 'column mis-match\n'
        msg += 'obtained columns\n'
        msg += f'{obtained.columns}\n'
        msg += 'expected columns\n'
        msg += f'{expected.columns}\n'

        missing_from_obtained = []
        for c in expected.columns:
            if c not in obtained.columns:
                missing_from_obtained.append(c)
        missing_from_expected = []
        for c in obtained.columns:
            if c not in expected.columns:
                missing_from_expected.append(c)
        msg += f'missing from obtained\n{missing_from_obtained}\n'
        msg += f'missing from expected\n{missing_from_expected}\n'
        raise RuntimeError(msg)

    if not expected.index.equals(obtained.index):
        msg += 'index mis-match\n'
        msg += 'expected index\n'
        msg += f'{expected.index}\n'
        msg += 'obtained index\n'
        msg += f'{obtained.index}\n'
        raise RuntimeError(msg)

    for col in expected.columns:
        expected_null = expected[col].isnull()
        obtained_null = obtained[col].isnull()
        if not expected_null.equals(obtained_null):
            msg += f'\n{col} not null at same point in '
            msg += 'obtained and expected\n'
            continue
        expected_valid = expected[~expected_null]
        obtained_valid = obtained[~obtained_null]
        if not expected_valid.index.equals(obtained_valid.index):
            msg += '\nindex mismatch in non-null when checking '
            msg += f'{col}\n'
        for index_val in expected_valid.index.values:
            e = expected_valid.at[index_val, col]
            o = obtained_valid.at[index_val, col]
            if isinstance(e, np.ndarray):
                e = list(e)
            if isinstance(o, np.ndarray):
                o = list(o)
            if not e == o:
                msg += f'\n{col}\n'
                msg += f'expected: {e}\n'
                msg += f'obtained: {o}\n'
    if msg != '':
        raise RuntimeError(msg)
