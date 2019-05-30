import pytest
from marshmallow import ValidationError
import os

from allensdk.brain_observatory.argschema_utilities import check_write_access #, check_write_access_overwrite

def write_some_text(path):
    with open(path, 'w') as fil:
        fil.write('some_text')

def existing_file(tmpdir):
    path = os.path.join(tmpdir, 'parent', 'foo.txt')
    os.makedirs(os.path.dirname(path))
    write_some_text(path)
    return path

def nonexistent_file(tmpdir):
    return os.path.join(tmpdir, 'parent', 'foo.txt')


@pytest.mark.parametrize('setup,raises', [
    [existing_file, True],
    [nonexistent_file, False]
])
def test_check_write_access(tmpdir_factory, setup, raises):

    # Definitely exists: base_dir
    base_dir = str(tmpdir_factory.mktemp('HW'))

    if raises:
        with pytest.raises(ValidationError):
            check_write_access(setup(base_dir))
    else:
        setup(base_dir)
