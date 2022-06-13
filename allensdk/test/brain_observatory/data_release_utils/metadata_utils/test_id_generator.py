import pytest
import pathlib
from allensdk.brain_observatory.data_release_utils \
    .metadata_utils.id_generator import (
        FileIDGenerator)


def test_not_a_file_error():
    """
    Test that an error is raised when you try to assign an ID to
    something that is not a file.
    """
    generator = FileIDGenerator()
    dummy = pathlib.Path('something.nwb')
    with pytest.raises(ValueError, match="is not a file"):
        generator.id_from_path(file_path=dummy)


def test_not_a_path_error(
        some_files_fixture):
    """
    Test that an error is raised when you try to get an ID from
    a file_path that is not a pathlib.Path
    """
    generator = FileIDGenerator()
    with pytest.raises(ValueError, match="must be a pathlib.Path"):
        str_path = str(some_files_fixture[1].resolve().absolute())
        generator.id_from_path(file_path=str_path)


def test_symlink_error(
        some_files_fixture,
        tmp_path_factory,
        helper_functions):
    """
    Test that an error is raised if you try to get an ID for a symlink
    """
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('symlink_test'))
    symlink_path = tmp_dir / 'a_silly_symlink.nwb'
    symlink_path.symlink_to(some_files_fixture[2])
    assert symlink_path.is_symlink()
    generator = FileIDGenerator()
    with pytest.raises(ValueError,
                       match="is a symlink; must be an actual path"):
        generator.id_from_path(file_path=symlink_path)
    helper_functions.windows_safe_cleanup(file_path=symlink_path)


def test_file_id_generator(
        some_files_fixture):
    """
    Test that FileIDGenerator assigns unique IDs to files
    """
    generator = FileIDGenerator()
    id_list = []
    for path in some_files_fixture:
        id_list.append(generator.id_from_path(file_path=path))

    # check that IDs are unique
    assert len(set(id_list)) == len(id_list)

    # check that IDs do not change
    for idx in range(len(some_files_fixture)):
        path = some_files_fixture[idx]
        expected = id_list[idx]
        assert generator.id_from_path(file_path=path) == expected
