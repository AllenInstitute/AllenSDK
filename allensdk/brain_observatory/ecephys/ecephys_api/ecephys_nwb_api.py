import warnings

import pandas as pd
import pynwb

from .ecephys_api import EcephysApi
from .. import RunningSpeed



class EcephysNwbApi(EcephysApi):

    __slots__ = ('path', '_nwbfile')

    @property
    def nwbfile(self):
        if hasattr(self, '_nwbfile'):
            return self._nwbfile

        io = pynwb.NWBHDF5IO(self.path, 'r')
        return io.read()

    def __init__(self, path, **kwargs):
        self.path = path

    def get_running_speed(self):
        return RunningSpeed(
            self.nwbfile.get_acquisition('running_speed').timestamps[:],
            self.nwbfile.get_acquisition('running_speed').data[:]
        )
    
    def get_stimulus_table(self):
        stimulus_table = self.nwbfile.epochs.to_dataframe()
        stimulus_table = stimulus_table.reset_index()
        stimulus_table.drop(columns=['tags', 'timeseries', 'id'], inplace=True)
        return stimulus_table
    
    def get_probes(self):
        probes = []
        for k, v in self.nwbfile.electrode_groups.items():
            probes.append({'id': int(k), 'description': v.description, 'location': v.location})
        probes = pd.DataFrame(probes)
        probes = probes.set_index(keys='id', drop=True)
        return probes
    
    def get_channels(self):
        channels = self.nwbfile.electrodes.to_dataframe()
        channels.drop(columns='group', inplace=True)
        return channels
    
    def get_mean_waveforms(self):
        units_table = self.__get_full_units_table()
        return units_table['waveform_mean'].to_dict()

    def get_spike_times(self):
        units_table = self.__get_full_units_table()
        return units_table['spike_times'].to_dict()
    
    def get_units(self):
        units_table = self.__get_full_units_table()
        units_table.drop(columns=['spike_times', 'waveform_mean'], inplace=True)

        return units_table

    def __get_full_units_table(self):
        return self.nwbfile.units.to_dataframe()

    @classmethod
    def from_nwbfile(cls, nwbfile, **kwargs):
        obj = cls(path=None, **kwargs)
        obj._nwbfile = nwbfile
        return obj