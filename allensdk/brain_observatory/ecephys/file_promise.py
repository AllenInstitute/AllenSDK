from typing import Dict, Union, List, Optional, Callable, Iterable, Any
from pathlib import Path

import pynwb

FileStream = Iterable[bytes]
FileStreamSource = Callable[[],FileStream]


def read_nwb(path):
    reader = pynwb.NWBHDF5IO(str(path), 'r')
    return reader.read()


def write_from_stream(path, stream):
    with open(path, "wb") as fil:
        for chunk in stream:
            fil.write(chunk)

class FilePromise:

    def __init__(self, source, path, reader):
        self.source: Optional[FileStreamSource] = source
        self.path: Path = Path(path)
        self.reader: Callable[Path, Any] = reader

    def __call__(self):
        if not self.path.exists():
            if self.source is None:
                raise FileNotFoundError(f"the file at {self.path} does not exist, and we don't know how to get it!")
            write_from_stream(self.path, self.source())

        return self.reader(self.path)