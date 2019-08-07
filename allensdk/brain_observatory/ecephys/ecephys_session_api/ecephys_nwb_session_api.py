from typing import Dict, Union, List, Optional, Callable, Iterable, Any
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import pynwb

from .ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.ecephys.file_promise import FilePromise
from allensdk.brain_observatory.nwb.nwb_api import NwbApi
import allensdk.brain_observatory.ecephys.nwb



class EcephysNwbSessionApi(NwbApi, EcephysSessionApi):

    def __init__(self, path, probe_lfp_paths: Optional[Dict[int, FilePromise]] = None, **kwargs):
        super(EcephysNwbSessionApi, self).__init__(path, **kwargs)
        self.probe_lfp_paths = probe_lfp_paths


    def get_session_start_time(self):
        return self.nwbfile.session_start_time


    def _probe_nwbfile(self, probe_id: int):
        if self.probe_lfp_paths is None:
            raise TypeError(
                f"EcephysNwbSessionApi assumes a split NWB file, with probewise LFP stored in individual files. "
                "this object was not configured with probe_lfp_paths"
            )
        elif probe_id not in self.probe_lfp_paths:
            raise KeyError(f"no probe lfp file path is recorded for probe {probe_id}")

        return self.probe_lfp_paths[probe_id]()


    def get_probes(self) -> pd.DataFrame:
        probes: Union[List, pd.DataFrame] = []
        for k, v in self.nwbfile.electrode_groups.items():
            probes.append({
                'id': int(k), 
                'description': v.description, 
                'location': v.location,
                "sampling_rate": v.sampling_rate,
                "lfp_sampling_rate": v.lfp_sampling_rate
            })
        probes = pd.DataFrame(probes)
        probes = probes.set_index(keys='id', drop=True)
        return probes

    def get_channels(self) -> pd.DataFrame:
        channels = self.nwbfile.electrodes.to_dataframe()
        channels.drop(columns='group', inplace=True)

        # these are stored as string in nwb 2, which is not ideal
        # float is also not ideal, but we have nans indicating out-of-brain structures
        channels["manual_structure_id"] = [float(chid) if chid != "" else np.nan for chid in channels["manual_structure_id"]]
        
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
        lfp_file = self._probe_nwbfile(probe_id)
        lfp = lfp_file.get_acquisition(f'probe_{probe_id}_lfp')
        series = lfp.get_electrical_series(f'probe_{probe_id}_lfp_data')

        electrodes = lfp_file.electrodes.to_dataframe()

        data = series.data[:]
        timestamps = series.timestamps[:]

        return xr.DataArray(
            name="LFP",
            data=data,
            dims=['time', 'channel'],
            coords=[timestamps, electrodes.index.values]
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
            running["net_rotation"] = rotation_series.data[:]

        return running

    def get_raw_running_data(self):
        rotation_series = self.nwbfile.get_acquisition("raw_running_wheel_rotation")
        signal_voltage_series = self.nwbfile.get_acquisition("running_wheel_signal_voltage")
        supply_voltage_series = self.nwbfile.get_acquisition("running_wheel_supply_voltage")

        return pd.DataFrame({
            "frame_time": rotation_series.timestamps[:],
            "net_rotation": rotation_series.data[:],
            "signal_voltage": signal_voltage_series.data[:],
            "supply_voltage": supply_voltage_series.data[:]
        })


    def get_ecephys_session_id(self) -> int:
        return int(self.nwbfile.identifier)


    def get_current_source_density(self, probe_id):
        csd_mod = self._probe_nwbfile(probe_id).get_processing_module("current_source_density")
        csd_ts = csd_mod["current_source_density"]

        return xr.DataArray(
            name="CSD",
            data=csd_ts.data[:],
            dims=["channel", "time"],
            coords={
                # Each CSD calculation step reduces the channel dimension by 2. 
                # These channels contributed to the calculation, 
                # but are not specifically associated with a row in the result
                "channel": csd_ts.control[2:-2], 
                "time": csd_ts.timestamps[:]
            }
        )


    def _get_full_units_table(self) -> pd.DataFrame:
        table = self.nwbfile.units.to_dataframe()
        table.index = table.index.astype(int)
        return table
