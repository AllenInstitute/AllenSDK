import pytest
from marshmallow import ValidationError
import os

from allensdk.brain_observatory.argschema_utilities import check_write_access #, check_write_access_overwrite


def test_check_write_access(tempdir_factory):

    # Definitely exists: base_dir
    base_dir = str(tempdir_factory.mktemp('HW'))
    filename_that_exists = os.path.join(base_dir, 'this_file_exists.txt')
    filename_that_exists_but_wrong_permissions = os.path.join(base_dir, 'this_file_exists.txt')
    for fname in [filename_that_exists, filename_that_exists_but_wrong_permissions]:
        with open(fname, 'w') as f:
            f.write('This string is now in the test file')

    # Now change permissions on filename_that_exists_but_wrong_permissions:




    # Definitely doesn't exist:
    dir_or_file_that_doesnt_exist = os.path.join(base_dir, 'impossible')

    # assert not check_write_access(dir_or_file_that_doesnt_exist)
    assert check_write_access(filename_that_exists)

    with pytest.raises(ValidationError) as e:
        check_write_access(dir_or_file_that_doesnt_exist)

    with pytest.raises(ValidationError) as e:
        check_write_access(filename_that_exists_but_wrong_permissions)

    
    

# def test_check_write_access_overwrite():



#     pass