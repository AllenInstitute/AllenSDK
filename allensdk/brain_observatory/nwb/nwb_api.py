import pynwb
from allensdk.brain_observatory.running_speed import RunningSpeed


class NwbApi:

    __slots__ = ('path', '_nwbfile')

    @property
    def nwbfile(self):
        if hasattr(self, '_nwbfile'):
            return self._nwbfile

        io = pynwb.NWBHDF5IO(self.path, 'r')
        return io.read()

    def __init__(self, path, **kwargs):
        ''' Reads data for a single Brain Observatory session from an NWB 2.0 file
        '''

        self.path = path

    @classmethod
    def from_nwbfile(cls, nwbfile, **kwargs):
        obj = cls(path=None, **kwargs)
        obj._nwbfile = nwbfile
        return obj

    def get_running_speed(self) -> RunningSpeed:

        values = self.nwbfile.modules['running'].get_data_interface('speed').data[:]
        timestamps = self.nwbfile.modules['running'].get_data_interface('timestamps').timestamps[:]

        return RunningSpeed(
            timestamps=timestamps,
            values=values,
        )
