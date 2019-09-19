from typing import Dict, Union, List, Optional, Callable, Iterable, Any
from pathlib import Path
from enum import IntEnum

import pandas as pd
import numpy as np
import xarray as xr
import pynwb

from .ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.ecephys.file_promise import FilePromise
from allensdk.brain_observatory.nwb.nwb_api import NwbApi
import allensdk.brain_observatory.ecephys.nwb
from allensdk.brain_observatory.ecephys import get_unit_filter_value


class EcephysNwbSessionApi(NwbApi, EcephysSessionApi):

    def __init__(self, path, probe_lfp_paths: Optional[Dict[int, FilePromise]] = None, **kwargs):

        self.filter_by_validity = kwargs.pop("filter_by_validity", True)
        self.amplitude_cutoff_maximum = get_unit_filter_value("amplitude_cutoff_maximum", **kwargs)
        self.presence_ratio_minimum = get_unit_filter_value("presence_ratio_minimum", **kwargs)
        self.isi_violations_maximum = get_unit_filter_value("isi_violations_maximum", **kwargs)

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
                "lfp_sampling_rate": v.lfp_sampling_rate,
                "has_lfp_data": v.has_lfp_data
            })
        probes = pd.DataFrame(probes)
        probes = probes.set_index(keys='id', drop=True)
        return probes

    def get_channels(self) -> pd.DataFrame:
        channels = self.nwbfile.electrodes.to_dataframe()
        channels.drop(columns='group', inplace=True)

        # these are stored as string in nwb 2, which is not ideal
        # float is also not ideal, but we have nans indicating out-of-brain structures
        channels["structure_id"] = [float(chid) if chid != "" else np.nan for chid in channels["manual_structure_id"]]
        channels.drop(columns="manual_structure_id")
        
        if self.filter_by_validity:
            channels = channels[channels["valid_data"]]
            channels = channels.drop(columns=["valid_data"])

        return channels

    def get_mean_waveforms(self) -> Dict[int, np.ndarray]:
        units_table = self._get_full_units_table()
        return units_table['waveform_mean'].to_dict()

    def get_spike_times(self) -> Dict[int, np.ndarray]:
        units_table = self._get_full_units_table()
        return units_table['spike_times'].to_dict()

    def get_spike_amplitudes(self) -> Dict[int, np.ndarray]:
        units_table = self._get_full_units_table()
        return units_table["spike_amplitudes"].to_dict()

    def get_units(self) -> pd.DataFrame:
        units = self._get_full_units_table()

        to_drop = set(["spike_times", "spike_amplitudes", "waveform_mean"]) & set(units.columns)
        units.drop(columns=list(to_drop), inplace=True)

        return units

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

    def get_eye_tracking_ellipse_fit_data(self):
        et_mod = self.nwbfile.get_processing_module("eye_tracking")

        et_data = {}
        et_data["cr_ellipse_fits"] = et_mod.get_data_interface("cr_ellipse_fits").to_dataframe()
        et_data["eye_ellipse_fits"] = et_mod.get_data_interface("eye_ellipse_fits").to_dataframe()
        et_data["pupil_ellipse_fits"] = et_mod.get_data_interface("pupil_ellipse_fits").to_dataframe()
        et_data["rig_geometry_data"] = et_mod.get_data_interface("rig_geometry_data").to_dataframe()
        rig_equipment = et_mod.get_data_interface("equipment").to_dataframe()
        et_data["rig_equipment"] = rig_equipment["equipment"][0]

        return et_data

    def get_raw_eye_gaze_mapping_data(self):
        rgm_mod = self.nwbfile.get_processing_module("raw_gaze_mapping")

        eye_area_ts = rgm_mod.get_data_interface("eye_area")
        pupil_area_ts = rgm_mod.get_data_interface("pupil_area")
        screen_coordinates_ts = rgm_mod.get_data_interface("screen_coordinates")
        screen_coordinates_spherical_ts = rgm_mod.get_data_interface("screen_coordinates_spherical")

        return pd.DataFrame({
            "frame_time": eye_area_ts.timestamps[:],
            "eye_area": eye_area_ts.data[:],
            "pupil_area": pupil_area_ts.data[:],
            "screen_coordinates_x_cm": screen_coordinates_ts.data[:, 1],
            "screen_coordinates_y_cm": screen_coordinates_ts.data[:, 0],
            "screen_coordinates_spherical_x_deg": screen_coordinates_spherical_ts.data[:, 1],
            "screen_coordinates_spherical_y_deg": screen_coordinates_spherical_ts.data[:, 0]
        })

    def get_filtered_eye_gaze_mapping_data(self):
        fgm_mod = self.nwbfile.get_processing_module("filtered_gaze_mapping")

        eye_area_ts = fgm_mod.get_data_interface("eye_area")
        pupil_area_ts = fgm_mod.get_data_interface("pupil_area")
        screen_coordinates_ts = fgm_mod.get_data_interface("screen_coordinates")
        screen_coordinates_spherical_ts = fgm_mod.get_data_interface("screen_coordinates_spherical")

        return pd.DataFrame({
            "frame_time": eye_area_ts.timestamps[:],
            "eye_area": eye_area_ts.data[:],
            "pupil_area": pupil_area_ts.data[:],
            "screen_coordinates_x_cm": screen_coordinates_ts.data[:, 1],
            "screen_coordinates_y_cm": screen_coordinates_ts.data[:, 0],
            "screen_coordinates_spherical_x_deg": screen_coordinates_spherical_ts.data[:, 1],
            "screen_coordinates_spherical_y_deg": screen_coordinates_spherical_ts.data[:, 0]
        })

    def get_ecephys_session_id(self) -> int:
        return int(self.nwbfile.identifier)


    def get_current_source_density(self, probe_id):
        csd_mod = self._probe_nwbfile(probe_id).get_processing_module("current_source_density")
        csd_ts = csd_mod["current_source_density"]

        csd = xr.DataArray(
            name="CSD",
            data=csd_ts.data[:],
            dims=["virtual_channel_index", "time"],
            coords={
                "virtual_channel_index": np.arange(csd_ts.data.shape[0]),
                "time": csd_ts.timestamps[:],
                "vertical_position": (("virtual_channel_index",), csd_ts.control[:, 1]),
                "horizontal_position": (("virtual_channel_index",), csd_ts.control[:, 0])
            }
        )
        return csd

    def get_optogenetic_stimulation(self) -> pd.DataFrame:
        mod = self.nwbfile.get_processing_module("optotagging")
        table = mod.get_data_interface("optogenetic_stimuluation").to_dataframe()
        table.drop(columns=["tags", "timeseries"], inplace=True)
        return table


    def _get_full_units_table(self) -> pd.DataFrame:
        units = self.nwbfile.units.to_dataframe()
        units.index = units.index.astype(int)

        if self.filter_by_validity:
            valid_channels = set(self.get_channels().index.values.tolist())
            units = units[
                (units["quality"] == "good")
                & (units["peak_channel_id"].isin(valid_channels))
            ]
            units.drop(columns=["quality"], inplace=True)

        units = units[units["amplitude_cutoff"] <= self.amplitude_cutoff_maximum]
        units = units[units["presence_ratio"] >= self.presence_ratio_minimum]
        units = units[units["isi_violations"] <= self.isi_violations_maximum]

        return units

    def get_metadata(self):
        return self.nwbfile.lab_meta_data['metadata'].to_dict()