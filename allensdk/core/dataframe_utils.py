from typing import List
import pandas as pd


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
