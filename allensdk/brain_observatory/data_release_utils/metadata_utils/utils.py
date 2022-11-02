import pandas as pd
import pathlib
import warnings

from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.id_generator import (
        FileIDGenerator)


def add_file_paths_to_metadata_table(
        metadata_table: pd.DataFrame,
        id_generator: FileIDGenerator,
        file_dir: pathlib.Path,
        file_prefix: str,
        index_col: str,
        session_id_col: str,
        on_missing_file: str) -> pd.DataFrame:
    """
    Add file_id and file_path columns to session dataframe.

    Parameters
    ----------
    metadata_table: pd.DataFrame
        The dataframe to which we are adding
        file_id and file_path

    id_generator: FileIDGenerator
        For maintaining a unique mapping between file_path and file_id

    file_dir: pathlib.Path
        directory where files will be found

    file_prefix: str
        Prefix of file names

    index_col: str
        Column in metadata_table used to index files

    session_id_col
        Column in metadata_table denoting session id

    on_missing_file: str
        Specifies how to handle missing files
            'error' -> raise an exception
            'warning' -> assign dummy file_id and warn
            'skip' -> drop that row from the table and warn

    Returns
    -------
    metadata_table:
        The same as the input dataframe but with file_id and file_path
        columns added

    Notes
    -----
    Files are assumed to be named like
    {file_dir}/{file_prefix}_{metadata_table.index_col}.nwb
    """

    if on_missing_file not in ('error', 'warn', 'skip'):
        msg = ("on_missing_file must be one of ('error', "
               "'warn', or 'skip'); you passed in "
               f"{on_missing_file}")
        raise ValueError(msg)

    file_suffix = 'nwb'
    new_data = []
    missing_files = []
    metadata_table = metadata_table.set_index(index_col)

    for row in metadata_table.itertuples():
        session_id = getattr(row, session_id_col, row.Index)
        file_path = file_dir / f'{session_id}' /  \
            f'{file_prefix}_{row.Index}.{file_suffix}'
        if not file_path.exists():
            file_id = id_generator.dummy_value
            missing_files.append(file_path.resolve().absolute())
        else:
            file_id = id_generator.id_from_path(file_path=file_path)
        str_path = str(file_path.resolve().absolute())
        new_data.append(
            {'file_id': file_id,
             'file_path': str_path,
             index_col: row.Index})

    if len(missing_files) > 0:
        msg = "The following files do not exist:"
        for file_path in missing_files:
            msg += f"\n{file_path}"
        if on_missing_file == 'error':
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

    new_df = pd.DataFrame(data=new_data)
    metadata_table = metadata_table.join(
                new_df.set_index(index_col),
                on=index_col,
                how='left')
    if on_missing_file == 'skip' and len(missing_files) > 0:
        metadata_table = metadata_table.drop(
            metadata_table.loc[
                metadata_table.file_id == id_generator.dummy_value].index)

    metadata_table = metadata_table.reset_index()

    return metadata_table
