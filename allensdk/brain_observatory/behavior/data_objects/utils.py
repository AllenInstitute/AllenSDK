from typing import List, Union
from collections import OrderedDict
import pandas


def create_empty_dataframe(
        column_names: List[str],
        index_name: Union[str, None]) -> pandas.DataFrame:
    """
    Create an empty dataframe to return in cases where
    a data object is missing from a session or experiment

    Parameters
    ----------
    column_names: List[str]
        List of names of the columns that should be in
        the DataFrame

    index_name: Union[str, None]
        Name of the DataFrame's index

    Returns
    -------
    pandas.DataFrame
    """
    data = OrderedDict()
    for column in column_names:
        data[column] = []
    df = pandas.DataFrame(data)
    if index_name is not None:
        df.index.name = index_name
    return df
