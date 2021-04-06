import pynwb
from pynwb import NWBFile


class NWBReader:
    def __init__(self, path):
        self._path = path

    def read(self) -> NWBFile:
        io = pynwb.NWBHDF5IO(self._path, 'r')
        return io.read()


class NWBWriter:
    def __init__(self, nwbfile: NWBFile, path: str):
        self._nwbfile = nwbfile
        self._path = path

    def write(self):
        with pynwb.NWBHDF5IO(self._path, 'w') as nwb_file_writer:
            nwb_file_writer.write(self._nwbfile)
