def get_column_name(table_cols: list,
                    possible_names: set) -> str:
    """
    This function acts a identifier for which column name is present in the
    dataframe from the provided possibilities. This is used in NWB to identify
    the correct column name for stimulus_name which differs from Behavior Ophys
    to Eccephys.
    :param table_cols: the table columns to search for the possible name within
    :param possible_names: the names that could exist within the data columns
    :return: the first entry of the intersection between the possible names
             and the names of the columns of the stimulus table
    """

    column_set = set(table_cols)
    column_names = list(column_set.intersection(possible_names))
    if not len(column_names) == 1:
        raise KeyError("Table expected one name column in intersection, found:"
                       f" {column_names}")
    return column_names[0]
