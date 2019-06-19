from typing import Dict, Union, List, Optional

import pandas as pd
import numpy as np
import xarray as xr
import pynwb

from .ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.nwb.nwb_api import NwbApi


class EcephysNwbSessionApi(NwbApi, EcephysSessionApi):

    def __init__(self, path, probe_lfp_paths: Optional[Dict[int, str]] = None, **kwargs):
        super(EcephysNwbSessionApi, self).__init__(path, **kwargs)
        self.probe_lfp_paths = probe_lfp_paths

    def _probe_nwbfile(self, probe_id: int):
        if self.probe_lfp_paths is None:
            raise TypeError(
                f"EcephysNwbSessionApi assumes a split NWB file, with probewise LFP stored in individual files. "
                "this object was not configured with probe_lfp_paths"
            )
        elif probe_id not in self.probe_lfp_paths:
            raise KeyError(f"no probe lfp file path is recorded for probe {probe_id}")

        reader = pynwb.NWBHDF5IO(self.probe_lfp_paths[probe_id], 'r')
        return reader.read()

    def get_probes(self) -> pd.DataFrame:
        probes: Union[List, pd.DataFrame] = []
        for k, v in self.nwbfile.electrode_groups.items():
            probes.append({'id': int(k), 'description': v.description, 'location': v.location})
        probes = pd.DataFrame(probes)
        probes = probes.set_index(keys='id', drop=True)
        probes['sampling_rate'] = 30000.0  # TODO: calculate real sampling rate for each probe.
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

    def get_lfp(self, probe_id: int) -> xr.DataArray:
        lfp = self._probe_nwbfile(probe_id).get_acquisition(f'probe_{probe_id}_lfp')
        series = lfp.get_electrical_series(f'probe_{probe_id}_lfp_data')

        electrodes = pd.DataFrame(
            data=[ecr for ecr in series.electrodes], 
            columns=['id'] + list(series.electrodes.table.colnames)
        )

        data = series.data[:]
        timestamps = series.timestamps[:]

        return xr.DataArray(
            data=data,
            dims=['time', 'channel'],
            coords=[timestamps, electrodes['id'].values]
        )

    def get_running_speed(self, include_rotation=False):
        running_module = self.nwbfile.get_processing_module("running")
        running_speed_series = running_module["running_speed"]

        running = pd.DataFrame({
            "start_time": running_speed_series.timestamps[0, :],
            "end_time": running_speed_series.timestamps[1, :],
            "velocity": running_speed_series.data[:]
        })

        if include_rotation:
            rotation_series = running_module["running_wheel_rotation"]
            running["rotation"] = rotation_series.data[:]

        return running

    def get_ecephys_session_id(self) -> int:
        return int(self.nwbfile.identifier)

    def _get_full_units_table(self) -> pd.DataFrame:
        table = self.nwbfile.units.to_dataframe()
        table.index = table.index.astype(int)
        return table
