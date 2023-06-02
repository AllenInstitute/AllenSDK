from typing import List
import pandas as pd

# Null value fill for integer Pandas.Series objects. NWB currently doesn't
# support using the new Int64 type that has explicit N/A values so we fill
# instead with -99.
INT_NULL = -99

"""A collection of utilities to manipulate pandas DataFrames."""


def patch_df_from_other(
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        columns_to_patch: List[str],
        index_column: str) -> pd.DataFrame:
    """
    Overwrite column values in target_df from column
    values in source_df in rows where the two dataframes
    share a value of index_column.

    Parameters
    ----------
    target_df: pd.DataFrame
        The dataframe whose columns will get overwritten

    source_df: pd.DataFrame
        The dataframe from which correct values are to be read

    columns_to_patch: List[str]
        The columns to be overwritten

    index_column: str
        The column to join the dataframes on

    Returns
    -------
    patched_df: pd.DataFrame
        target_df except with the specified columns and rows
        overwritten.

    Notes
    -----
    If any of the columns_to_patch are not in target_df, they
    will be added.

    This function starts by creating a copy of target_df, so
    it will not alter the argument in-place.
    """
    target_df = target_df.copy(deep=True)
    original_index = target_df.index.name
    if original_index is not None:
        target_df = target_df.reset_index()

    msg = ""
    if index_column not in target_df.columns:
        msg += f"{index_column} not in target_df\n"

    if index_column not in source_df.columns:
        msg += f"{index_column} not in source_df\n"
    else:
        index_values = source_df[index_column].values
        if len(set(index_values)) != len(index_values):
            msg += f"{index_column} values in source_df are not unique\n"

    for column in columns_to_patch:
        if column not in source_df:
            msg += f"{column} not in source_df\n"
        if column not in target_df:
            target_df[column] = None

    if index_column in columns_to_patch:
        msg += (f"{index_column} is in the list of "
                f"columns to patch {columns_to_patch}; "
                "unsure how to handle that case\n")

    if len(msg) > 0:
        msg = f"failures in patch_df_from_other:\n{msg}"
        raise ValueError(msg)

    target_df = target_df.set_index(index_column)

    patch_df = source_df[columns_to_patch + [index_column]]
    patch_df = patch_df.set_index(index_column)

    target_df.update(
        patch_df,
        join='left',
        overwrite=True)

    target_df = target_df.reset_index()
    if original_index is not None:
        target_df = target_df.set_index(original_index)
    return target_df


def enforce_df_column_order(
        input_df: pd.DataFrame,
        column_order: List[str]) -> pd.DataFrame:
    """Return the data frame but with columns ordered.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Data frame with columns to be ordered.
    column_order : list of str
        Ordering of column names to enforce. Columns not specified are shifted
        to the end of the order but retain their order amongst others not
        specified. If a specified column is not in the DataFrame it is ignored.

    Returns
    -------
    output_df : pandas.DataFrame
        DataFrame the same as the input but with columns reordered.
    """
    # Use only columns that are in the input dataframe's columns.
    pruned_order = []
    for col in column_order:
        if col in input_df.columns:
            pruned_order.append(col)
    # Get the full list of columns in the data frame with our ordered columns
    # first.
    pruned_order.extend(
        list(set(input_df.columns).difference(set(pruned_order))))
    return input_df[pruned_order]


def enforce_df_int_typing(input_df: pd.DataFrame,
                          int_columns: List[str]) -> pd.DataFrame:
    """Enforce integer typing for columns that may have lost int typing when
    combined into the final DataFrame.

    Parameters
    ----------
    input_df : pandas.DataFrame
        DataFrame with typing to enforce.
    int_columns : list of str
        Columns to enforce int typing and fill any NaN/None values with the
        value set in INT_NULL in this file. Requested columns not in the
        dataframe are ignored.

    Returns
    -------
    output_df : pandas.DataFrame
        DataFrame specific columns hard typed to Int64 to allow NA values
        without resorting to float type.
    """
    for col in int_columns:
        if col in input_df.columns:
            input_df[col] = \
                input_df[col].fillna(INT_NULL).astype(int)
    return input_df
