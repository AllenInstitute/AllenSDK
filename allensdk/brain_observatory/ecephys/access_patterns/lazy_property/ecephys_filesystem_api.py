from collections import namedtuple
import random
import warnings

import pandas as pd
import pynwb

from ..ecephys_api import EcephysApi


RunningSpeed = namedtuple(typename='RunningSpeed', field_names=('timestamps', 'values'))


class EcephysFilesystemApi(EcephysApi):
    
    __slots__ = ('io', 'nwbfile', '__hidden_units')

    def get_hidden_units(self, available):
        if self.__hidden_units == 'no_hidden_units':
            return set()
        elif self.__hidden_units is None:
            self.__hidden_units = set(random.choices(available, k=20))
        return self.__hidden_units

    def __init__(self, path, randomly_hide_units=False, **kwargs):
        self.io = pynwb.NWBHDF5IO(path, 'r')
        self.nwbfile = self.io.read()

        if randomly_hide_units:
            self.__hidden_units = None
        else:
            self.__hidden_units = 'no_hidden_units'

    def get_running_speed(self):
        return RunningSpeed(
            self.nwbfile.get_acquisition('running_speed').timestamps[:],
            self.nwbfile.get_acquisition('running_speed').data[:]
        )
    
    def get_stimulus_table(self):
        stimulus_table = self.nwbfile.epochs.to_dataframe()
        stimulus_table.drop(columns=['tags', 'timeseries'], inplace=True)
        return stimulus_table
    
    def get_probes(self):
        probes = []
        for k, v in self.nwbfile.electrode_groups.items():
            probes.append({'id': int(k), 'name': v.description})
        return pd.DataFrame(probes)
    
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
        units_table = self.nwbfile.units.to_dataframe()

        hidden_units = self.get_hidden_units(units_table.index.values)
        units_table = units_table.loc[~units_table.index.isin(hidden_units), :]

        return units_table