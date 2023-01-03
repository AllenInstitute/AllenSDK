import pytest
import pandas as pd

from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.id_generator import (
        FileIDGenerator)

from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.utils import (
        add_file_paths_to_metadata_table)


def test_add_file_paths_to_metadata_table_on_missing_error(
        some_files_fixture,
        metadata_table_fixture):
    """
    Test that an error is raised by add_file_paths_to_metadata_table
    when on_missing_file is a nonsense value
    """
    with pytest.raises(ValueError,
                       match="on_missing_file must be one of"):
        add_file_paths_to_metadata_table(
            metadata_table=metadata_table_fixture,
            id_generator=FileIDGenerator(),
            file_dir=some_files_fixture[0].parent,
            file_prefix='silly_file',
            index_col='file_index',
            on_missing_file='whatever',
            data_dir_col='session_id'
        )


def test_add_file_paths_to_metadata_table_no_file_error(
        some_files_fixture,
        metadata_table_fixture):
    """
    Test that an error is raised by add_file_paths_to_metadata_table
    when files are missing (if requested)
    """
    with pytest.raises(RuntimeError,
                       match="The following files do not exist"):
        add_file_paths_to_metadata_table(
            metadata_table=metadata_table_fixture,
            id_generator=FileIDGenerator(),
            file_dir=some_files_fixture[0].parent,
            file_prefix='silly_file',
            index_col='file_index',
            on_missing_file='error',
            data_dir_col='session_id'
        )


@pytest.mark.parametrize(
        'on_missing_file', ['skip', 'warn'])
def test_add_file_paths_to_metadata_table(
        some_files_fixture,
        metadata_table_fixture,
        on_missing_file):
    """
    Test that add_file_paths_to_metadata_table behaves as expected
    when not raising an error
    """
    file_dir = some_files_fixture[0].parent.parent
    expected = metadata_table_fixture.copy(deep=True)

    id_generator = FileIDGenerator()

    with pytest.warns(UserWarning,
                      match='The following files do not exist'):
        result = add_file_paths_to_metadata_table(
            metadata_table=metadata_table_fixture,
            id_generator=id_generator,
            file_dir=file_dir,
            file_prefix='silly_file',
            index_col='file_index',
            on_missing_file=on_missing_file,
            data_dir_col='session_id'
        )

    # because we have not yet added file_id and file_path
    # to expected
    assert not expected.equals(result)

    if on_missing_file == 'skip':
        assert len(result) == len(some_files_fixture)
    else:
        assert len(result) == len(some_files_fixture) + 2

    expected['file_id'] = id_generator.dummy_value
    expected['file_path'] = 'nothing'

    for file_idx in expected.file_index:
        file_path = file_dir / f'{file_idx}' / f'silly_file_{file_idx}.nwb'
        str_path = str(file_path.resolve().absolute())
        expected.loc[expected.file_index == file_idx, 'file_path'] = str_path
        if file_path.exists():
            file_id = id_generator.id_from_path(file_path=file_path)
            expected.loc[expected.file_index == file_idx, 'file_id'] = file_id
        elif on_missing_file == 'skip':
            expected = expected.drop(
                    expected.loc[expected.file_index == file_idx].index)

    pd.testing.assert_frame_equal(result, expected)
