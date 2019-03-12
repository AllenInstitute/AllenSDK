import warnings
from typing import Dict, Union, List

import pandas as pd
import numpy as np
import pynwb

from .ecephys_api import EcephysApi
from ... import RunningSpeed



class EcephysNwbApi(EcephysApi):

    __slots__ = ('path', '_nwbfile')

    @property
    def nwbfile(self):
        if hasattr(self, '_nwbfile'):
            return self._nwbfile

        io = pynwb.NWBHDF5IO(self.path, 'r')
        return io.read()

    def __init__(self, path, **kwargs):
        ''' Reads data for a single Extracellular Electrophysiology session from an NWB 2.0 file
        '''

        self.path = path

    def get_running_speed(self) -> RunningSpeed:
        return RunningSpeed(
            timestamps=self.nwbfile.get_acquisition('running_speed').timestamps[:],
            values=self.nwbfile.get_acquisition('running_speed').data[:]
        )
    
    def get_stimulus_table(self) -> pd.DataFrame:
        table = pd.DataFrame({
            col.name: col.data for col in self.nwbfile.epochs.columns 
            if col.name not in set(['tags', 'timeseries', 'tags_index', 'timeseries_index'])
        }, index=pd.Index(name=self.nwbfile.epochs.id.name, data=self.nwbfile.epochs.id.data))
        table.index = table.index.astype(int)
        return table
    
    
    def get_probes(self) -> pd.DataFrame:
        probes: Union[List, pd.DataFrame] = []
        for k, v in self.nwbfile.electrode_groups.items():
            probes.append({'id': int(k), 'description': v.description, 'location': v.location})
        probes = pd.DataFrame(probes)
        probes = probes.set_index(keys='id', drop=True)
        return probes
    
    def get_channels(self) -> pd.DataFrame:
        channels = self.nwbfile.electrodes.to_dataframe()
        channels.drop(columns='group', inplace=True)
        return channels
    
    def get_mean_waveforms(self) -> Dict[int, np.ndarray]:
        units_table = self._get_full_units_table()
        return units_table['waveform_mean'].to_dict()

    def get_spike_times(self) -> Dict[int, np.ndarray]:
        units_table = self._get_full_units_table()
        return units_table['spike_times'].to_dict()
    
    def get_units(self) -> pd.DataFrame:
        units_table = self._get_full_units_table()
        units_table.drop(columns=['spike_times', 'waveform_mean'], inplace=True)

        return units_table

    def _get_full_units_table(self) -> pd.DataFrame:
        table = self.nwbfile.units.to_dataframe()
        table.index = table.index.astype(int)
        return table

    @classmethod
    def from_nwbfile(cls, nwbfile, **kwargs):
        obj = cls(path=None, **kwargs)
        obj._nwbfile = nwbfile
        return obj