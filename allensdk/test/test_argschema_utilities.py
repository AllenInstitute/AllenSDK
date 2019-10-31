import os
import stat
from pathlib import Path

import pytest
from allensdk.brain_observatory.argschema_utilities import (
    InputFile,
    OutputFile,
    RaisingSchema,
    check_write_access,
    check_write_access_overwrite)
from marshmallow import Schema, ValidationError

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


class GenericInputSchema(Schema):
    input_file = InputFile(required=True)


class GenericOutputSchema(RaisingSchema):
    output_file = OutputFile(required=True)


class TestInputFile(object):

    def setup_method(self):
        self.parser = GenericInputSchema()

    @pytest.mark.parametrize("input_data", [
        ({"input_file": "/some/invalid_filepath/input.h5"}),
    ])
    def test_invalid_input_file(self, input_data):
        with pytest.raises(ValidationError, match=r"No such file or directory"):
            self.parser.load(input_data)

    def test_valid_input_file(self, tmpdir):
        p = tmpdir.mkdir("input_file").join("valid_input.h5")
        p.write("stuff")

        path_str = str(p)
        obtained = self.parser.load({"input_file": path_str})
        assert obtained["input_file"] == Path(path_str)


class TestOutputFile(object):

    def setup_method(self):
        self.parser = GenericOutputSchema()

    @pytest.mark.parametrize("output_data", [
        ({"output_file": "////invalid_filepath/output.json"}),
    ])
    def test_invalid_output_file(self, output_data):
        # Apparently allensdk.brain_observatory.argschema_utilities tests are
        # skipped on Windows systems and the `check_write_access_overwrite`
        # function itself does not work correctly on Windows systems.
        # TODO: This is a stopgap for now
        if os.name == 'nt':
            pytest.skip()
        # This test was failing on Bamboo because it was run in a container
        # as root (which means pretty much anything is writable). If this test
        # is run as root, skip it as it will always successfully create the
        # output file.
        if os.getuid() == 0:
            pytest.skip()

        with pytest.raises(ValidationError, match="Can't build path to requested location"):
            self.parser.load(output_data)

    def test_valid_output_file(self, tmpdir):
        p = tmpdir.mkdir("output_file").join("output_file.json")

        path_str = str(p)
        obtained = self.parser.load({"output_file": path_str})
        assert obtained["output_file"] == Path(path_str)
