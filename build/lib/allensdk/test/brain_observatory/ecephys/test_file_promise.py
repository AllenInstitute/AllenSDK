from pathlib import Path

import pytest

from allensdk.brain_observatory.ecephys.file_promise import FilePromise


def test_file_promise(tmpdir_factory):

    tmpdir = Path(tmpdir_factory.mktemp("test_file_promise"))
    path = tmpdir / Path("note.txt")

    data = [b"hello", b"world"]
    source = lambda *a, **k: (element for element in data)

    def reader(p):
        with open(p, "rb") as f:
            return f.read()

    promise = FilePromise(source, path, reader)

    assert promise().decode() == "helloworld"
    assert path.exists()