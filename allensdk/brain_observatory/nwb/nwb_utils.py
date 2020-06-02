def get_stimulus_name_column(stimulus_table_cols: list,
                             possible_names: set) -> str:
    """
    This function acts a identifier for which column name is present in the
    dataframe to indict the name of a stimuli. In behavior ophys sessions this
    is 'image_name' and in eceephys this is 'stimulus_name' as an example
    where the NWB write functions use the same function with a name change.
    :param stimulus_table_cols: the table columns to search for the stimulus
                                name within
    :param possible_names: the names that could exist within the data columns
    :return: the first entry of the intersection between the possible names
             and the names of the columns of the stimulus table
    """

    stimulus_column_set = set(stimulus_table_cols)
    stim_column_names = list(stimulus_column_set.intersection(possible_names))
    if not len(stim_column_names) == 1:
        raise KeyError("Stimulus table does not have correct names for stimulus"
                       " name column, expected one name in intersection, found:"
                       f" {stim_column_names}")
    return stim_column_names[0]
