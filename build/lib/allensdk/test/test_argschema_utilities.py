import pytest
from marshmallow import ValidationError
import os
import stat
import platform

from allensdk.brain_observatory.argschema_utilities import (
    check_write_access,
    check_write_access_overwrite,
)

READ_ONLY = stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH
READ_WRITE = READ_ONLY | stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH


def write_some_text(path):
    with open(path, "w") as fil:
        fil.write("some_text")


def try_write_bad_permissions(path):
    try:
        p = os.path.join(path, "check")
        open(p, "w")
        pytest.skip()
    except PermissionError:
        pass


class WriteAccessTestHarness(object):
    def __init__(self, base_path):
        self.base_path = base_path

    def setup(self):
        raise NotImplementedError()

    def teardown(self):
        pass


class ExistingFile(WriteAccessTestHarness):
    def setup(self):
        self.path = os.path.join(self.base_path, "parent", "foo")
        os.makedirs(os.path.dirname(self.path))
        write_some_text(self.path)
        return self.path


class NonexistentFile(WriteAccessTestHarness):
    def setup(self):
        self.path = os.path.join(self.base_path, "parent", "nonexistent_file.txt")
        return self.path


class FileInBadPermissionsDir(WriteAccessTestHarness):
    def setup(self):
        self.first = os.path.join(self.base_path, "no_write")

        os.makedirs(self.first)
        os.chmod(self.first, READ_ONLY)

        try_write_bad_permissions(self.first)

        self.path = os.path.join(self.first, "foo.txt")
        return self.path

    def teardown(self):
        os.chmod(self.first, READ_WRITE)


class FileInBadPermissionsMiddleDir(WriteAccessTestHarness):
    def setup(self):
        self.first = os.path.join(self.base_path, "first")
        self.second = os.path.join(self.first, "second")

        os.makedirs(self.first)
        os.chmod(self.first, READ_ONLY)

        try_write_bad_permissions(self.first)

        self.path = os.path.join(self.second, "foo.txt")
        return self.path

    def teardown(self):
        os.chmod(self.first, READ_WRITE)


@pytest.mark.parametrize(
    "harness_cls,fn,raises",
    [
        [ExistingFile, check_write_access, True],
        [ExistingFile, check_write_access_overwrite, False],
        [NonexistentFile, check_write_access, False],
        [NonexistentFile, check_write_access_overwrite, False],
        [FileInBadPermissionsDir, check_write_access, True],
        [FileInBadPermissionsDir, check_write_access_overwrite, True],
        [FileInBadPermissionsMiddleDir, check_write_access, True],
        [FileInBadPermissionsMiddleDir, check_write_access_overwrite, True],
    ],
)
def test_check_write_access(tmpdir_factory, harness_cls, fn, raises):

    base_dir = str(tmpdir_factory.mktemp("HW"))

    harness = harness_cls(base_dir)
    testpath = harness.setup()

    if raises:
        with pytest.raises(ValidationError):
            fn(testpath)
    else:
        assert fn(testpath)

    harness.teardown()
