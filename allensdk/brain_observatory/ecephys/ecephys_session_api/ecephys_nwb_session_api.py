from typing import Dict, Union, List, Optional, Callable, Iterable, Any
from pathlib import Path
import re
import ast

import pandas as pd
import numpy as np
import xarray as xr
import pynwb

from .ecephys_session_api import EcephysSessionApi
from allensdk.brain_observatory.nwb.nwb_api import NwbApi
import allensdk.brain_observatory.ecephys.nwb  # noqa Necessary to import pyNWB namespaces
from allensdk.brain_observatory.ecephys import get_unit_filter_value


color_triplet_re = re.compile(r"\[(-{0,1}\d*\.\d*,\s*)*(-{0,1}\d*\.\d*)\]")


class EcephysNwbSessionApi(NwbApi, EcephysSessionApi):

    def __init__(self,
                 path,
                 probe_lfp_paths: Optional[Dict[int, Callable[[], pynwb.NWBFile]]] = None,
                 additional_unit_metrics=None,
                 external_channel_columns=None,
                 **kwargs):

        self.filter_out_of_brain_units = kwargs.pop("filter_out_of_brain_units", True)
        self.filter_by_validity = kwargs.pop("filter_by_validity", True)
        self.amplitude_cutoff_maximum = get_unit_filter_value("amplitude_cutoff_maximum", **kwargs)
        self.presence_ratio_minimum = get_unit_filter_value("presence_ratio_minimum", **kwargs)
        self.isi_violations_maximum = get_unit_filter_value("isi_violations_maximum", **kwargs)

        super(EcephysNwbSessionApi, self).__init__(path, **kwargs)
        self.probe_lfp_paths = probe_lfp_paths

        self.additional_unit_metrics = additional_unit_metrics
        self.external_channel_columns = external_channel_columns

    def test(self):
        """ A minimal test to make sure that this API's NWB file exists and is 
        readable. Ecephys NWB files use the required session identifier field 
        to store the session id, so this is guaranteed to be present for any 
        uncorrupted NWB file.

        Of course, this does not ensure that the file as a whole is correct.
        """
        self.get_ecephys_session_id()

    def get_session_start_time(self):
        return self.nwbfile.session_start_time

    def get_stimulus_presentations(self):
        table = super(EcephysNwbSessionApi, self).get_stimulus_presentations()
        
        if "color" in table.columns:
            # the color column actually contains two parameters. One is coded as rgb triplets and the other as -1 or 1
            if "color_triplet" not in table.columns:
                table["color_triplet"] = pd.Series("", index=table.index)
            rgb_color_match = table["color"].str.match(color_triplet_re)
            table.loc[rgb_color_match, "color_triplet"] = table.loc[rgb_color_match, "color"]
            table.loc[rgb_color_match, "color"] = ""

            # make sure the color column's values are numeric
            table.loc[table["color"] != "", "color"] = table.loc[table["color"] != "", "color"].apply(ast.literal_eval)

        return table

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

        # Rename columns for clarity
        channels.rename(
            columns={"manual_structure_id": "ecephys_structure_id",
                     "manual_structure_acronym": "ecephys_structure_acronym"},
            inplace=True)

        # these are stored as string in nwb 2, which is not ideal
        # float is also not ideal, but we have nans indicating out-of-brain structures
        channels["ecephys_structure_id"] = [
            float(chid) if chid != "" 
            else np.nan 
            for chid in channels["ecephys_structure_id"]
        ]
        channels["ecephys_structure_acronym"] = [
            ch_acr if ch_acr not in set(["None", ""]) 
            else np.nan 
            for ch_acr in channels["ecephys_structure_acronym"]
        ]

        if self.external_channel_columns is not None:
            external_channel_columns = self.external_channel_columns()
            channels = clobbering_merge(channels, external_channel_columns, left_index=True, right_index=True)

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

        if self.additional_unit_metrics is not None:
            additional_metrics = self.additional_unit_metrics()
            units = pd.merge(units, additional_metrics, left_index=True, right_index=True)

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

    def get_rig_metadata(self) -> Optional[dict]:
        try:
            et_mod = self.nwbfile.get_processing_module("eye_tracking")
        except KeyError as e:
            print(f"This ecephys session '{int(self.nwbfile.identifier)}' has no eye tracking data (and thus no rig geometry data). (NWB error: {e})")
            return None

        rig_metadata = {}
        rig_metadata["rig_geometry_data"] = et_mod.get_data_interface("rig_geometry_data").to_dataframe()
        rig_equipment = et_mod.get_data_interface("equipment").to_dataframe()
        rig_metadata["rig_equipment"] = rig_equipment["equipment"][0]

        return rig_metadata

    def get_pupil_data(self, suppress_pupil_data: bool = True) -> Optional[pd.DataFrame]:
        try:
            et_mod = self.nwbfile.get_processing_module("eye_tracking")
            rgm_mod = self.nwbfile.get_processing_module("raw_gaze_mapping")
            fgm_mod = self.nwbfile.get_processing_module("filtered_gaze_mapping")
        except KeyError as e:
            print(f"This ecephys session '{int(self.nwbfile.identifier)}' has no eye tracking data. (NWB error: {e})")
            return None

        raw_eye_area_ts = rgm_mod.get_data_interface("eye_area")
        raw_pupil_area_ts = rgm_mod.get_data_interface("pupil_area")
        raw_screen_coordinates_ts = rgm_mod.get_data_interface("screen_coordinates")
        raw_screen_coordinates_spherical_ts = rgm_mod.get_data_interface("screen_coordinates_spherical")

        filtered_eye_area_ts = fgm_mod.get_data_interface("eye_area")
        filtered_pupil_area_ts = fgm_mod.get_data_interface("pupil_area")
        filtered_screen_coordinates_ts = fgm_mod.get_data_interface("screen_coordinates")
        filtered_screen_coordinates_spherical_ts = fgm_mod.get_data_interface("screen_coordinates_spherical")

        cr_ellipse_fits = et_mod.get_data_interface("cr_ellipse_fits").to_dataframe()
        eye_ellipse_fits = et_mod.get_data_interface("eye_ellipse_fits").to_dataframe()
        pupil_ellipse_fits = et_mod.get_data_interface("pupil_ellipse_fits").to_dataframe()

        eye_tracking_data = {
            "corneal_reflection_center_x": cr_ellipse_fits["center_x"].values,
            "corneal_reflection_center_y": cr_ellipse_fits["center_y"].values,
            "corneal_reflection_height": cr_ellipse_fits["height"].values,
            "corneal_reflection_width": cr_ellipse_fits["width"].values,
            "corneal_reflection_phi": cr_ellipse_fits["phi"].values,

            "pupil_center_x": pupil_ellipse_fits["center_x"].values,
            "pupil_center_y": pupil_ellipse_fits["center_y"].values,
            "pupil_height": pupil_ellipse_fits["height"].values,
            "pupil_width": pupil_ellipse_fits["width"].values,
            "pupil_phi": pupil_ellipse_fits["phi"].values,

            "eye_center_x": eye_ellipse_fits["center_x"].values,
            "eye_center_y": eye_ellipse_fits["center_y"].values,
            "eye_height": eye_ellipse_fits["height"].values,
            "eye_width": eye_ellipse_fits["width"].values,
            "eye_phi": eye_ellipse_fits["phi"].values
        }

        if not suppress_pupil_data:
            eye_tracking_data.update(
                {
                    "raw_eye_area": raw_eye_area_ts.data[:],
                    "raw_pupil_area": raw_pupil_area_ts.data[:],
                    "raw_screen_coordinates_x_cm": raw_screen_coordinates_ts.data[:, 1],
                    "raw_screen_coordinates_y_cm": raw_screen_coordinates_ts.data[:, 0],
                    "raw_screen_coordinates_spherical_x_deg": raw_screen_coordinates_spherical_ts.data[:, 1],
                    "raw_screen_coordinates_spherical_y_deg": raw_screen_coordinates_spherical_ts.data[:, 0],

                    "filtered_eye_area": filtered_eye_area_ts.data[:],
                    "filtered_pupil_area": filtered_pupil_area_ts.data[:],
                    "filtered_screen_coordinates_x_cm": filtered_screen_coordinates_ts.data[:, 1],
                    "filtered_screen_coordinates_y_cm": filtered_screen_coordinates_ts.data[:, 0],
                    "filtered_screen_coordinates_spherical_x_deg": filtered_screen_coordinates_spherical_ts.data[:, 1],
                    "filtered_screen_coordinates_spherical_y_deg": filtered_screen_coordinates_spherical_ts.data[:, 0]
                }
            )

        return pd.DataFrame(eye_tracking_data, index=raw_eye_area_ts.timestamps[:])

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

        if self.filter_by_validity or self.filter_out_of_brain_units:
            channels = self.get_channels()

            if self.filter_out_of_brain_units:
                channels = channels[~(channels["ecephys_structure_id"].isna())]

            channel_ids = set(channels.index.values.tolist())
            units = units[units["peak_channel_id"].isin(channel_ids)]

        if self.filter_by_validity:
            units = units[units["quality"] == "good"]
            units.drop(columns=["quality"], inplace=True)

        units = units[units["amplitude_cutoff"] <= self.amplitude_cutoff_maximum]
        units = units[units["presence_ratio"] >= self.presence_ratio_minimum]
        units = units[units["isi_violations"] <= self.isi_violations_maximum]
        return units

    def get_metadata(self):
        return self.nwbfile.lab_meta_data['metadata'].to_dict()


def clobbering_merge(to_df, from_df, **kwargs):
    overlapping = set(to_df.columns) & set(from_df.columns)
    
    for merge_param in ["on", "left_on", "right_on"]:
        if merge_param in kwargs:
            merge_arg = kwargs.get(merge_param)
            if isinstance(merge_arg, str):
                merge_arg = [merge_arg]
            overlapping = overlapping - set(list(merge_arg))

    to_df = to_df.drop(columns=list(overlapping))
    return pd.merge(to_df, from_df, **kwargs)
