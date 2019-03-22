import os
import pandas as pd
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

    @classmethod
    def from_path(cls, path, **kwargs):

        try:
            with open(path, 'r'):
                pass
        except Exception as err:
            raise

        return cls(path=path)

    def get_running_speed(self) -> RunningSpeed:

        values = self.nwbfile.modules['running'].get_data_interface('speed').data[:]
        timestamps = self.nwbfile.modules['running'].get_data_interface('timestamps').timestamps[:]

        return RunningSpeed(
            timestamps=timestamps,
            values=values,
        )

    def get_stimulus_presentations(self) -> pd.DataFrame:
        table = pd.DataFrame({
            col.name: col.data for col in self.nwbfile.epochs.columns 
            if col.name not in set(['tags', 'timeseries', 'tags_index', 'timeseries_index'])
        }, index=pd.Index(name='stimulus_presentations_id', data=self.nwbfile.epochs.id.data))
        table.index = table.index.astype(int)
        return table
