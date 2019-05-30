import pytest
from marshmallow import ValidationError
import os
import stat

from allensdk.brain_observatory.argschema_utilities import check_write_access #, check_write_access_overwrite


def write_some_text(path):
    with open(path, 'w') as fil:
        fil.write('some_text')


def existing_file(tmpdir_factory, filename='existing_file.txt'):
    base_dir = str(tmpdir_factory.mktemp('HW'))
    path = os.path.join(base_dir, 'parent', filename)
    os.makedirs(os.path.dirname(path))
    write_some_text(path)
    return path


def file_in_bad_permissions_dir(tmpdir_factory):
    base_dir = str(tmpdir_factory.mktemp('HW'))
    os.chmod(base_dir, stat.S_IREAD)
    return os.path.join(base_dir, 'parent_doesnt_have_permission_file.txt')


def file_in_bad_permissions_middle_dir(tmpdir_factory):
    base_dir = str(tmpdir_factory.mktemp('HW'))
    os.chmod(base_dir, stat.S_IREAD)
    return os.path.join(base_dir, 'doesnt_exist', 'parent_doesnt_have_permission_file.txt')


def nonexistent_file(tmpdir_factory):
    base_dir = str(tmpdir_factory.mktemp('HW'))
    return os.path.join(base_dir, 'parent', 'nonexistent_file.txt')


@pytest.mark.parametrize('setup,raises', [
    [existing_file, True],
    [nonexistent_file, False],
    [file_in_bad_permissions_dir, True],
    [file_in_bad_permissions_middle_dir, True]
])
def test_check_write_access(tmpdir_factory, setup, raises):

    testpath = setup(tmpdir_factory)
    if raises:
        with pytest.raises(ValidationError):
            check_write_access(testpath)
    else:
        assert check_write_access(testpath)
