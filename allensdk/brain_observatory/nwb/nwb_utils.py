import pandas as pd
# All of the omitted stimuli have a duration of 250ms as defined
# by the Visual Behavior team. For questions about duration contact that
# team.
omitted_stimuli_duration = 0.250


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


def set_omitted_stop_time(stimulus_table: pd.DataFrame) -> None:
    """
    This function sets the stop time for a row that of a stimuli table that
    is a omitted stimuli. A omitted stimuli is a stimuli where a mouse is
    shown only a grey screen and these last for 250 milliseconds. These do not
    include a stop_time or end_frame as other stimuli in the stimulus table due
    to design choices. For these stimuli to be added they must have the
    stop_time calculated and put into the row as data before writing to NWB.
    :param stimulus_table: pd.DataFrame that contains the stimuli presented to
                           an experiment subject
    :return:
          stimulus_table_row: returns the same dictionary as inputted but with
                              an additional entry for stop_time.
    """
    omitted_row_indexs = stimulus_table.index[stimulus_table['omitted']].tolist()
    for omitted_row_idx in omitted_row_indexs:
        row = stimulus_table.iloc[omitted_row_idx]
        start_time = row['start_time']
        end_time = start_time + omitted_stimuli_duration
        row['stop_time'] = end_time
        row['duration'] = omitted_stimuli_duration
        stimulus_table.iloc[omitted_row_idx] = row