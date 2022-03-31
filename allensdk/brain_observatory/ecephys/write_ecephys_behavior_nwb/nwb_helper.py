import pynwb
from pynwb import NWBFile
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from allensdk.brain_observatory import dict_to_indexed_array

# from allensdk.brain_observatory.ecephys.write_nwb.__main__ import add_probewise_data_to_nwbfile
from allensdk.brain_observatory.ecephys.write_nwb.__main__ import (
  add_probewise_data_to_nwbfile,
  read_running_speed,
  add_running_speed_to_nwbfile,
  add_raw_running_data_to_nwbfile,
  add_optotagging_table_to_nwbfile,
  add_eye_tracking_rig_geometry_data_to_nwbfile,
  read_stimulus_table,
  add_stimulus_timestamps,
  add_stimulus_presentations
)

from allensdk.brain_observatory.ecephys.nwb import (
    EcephysProbe,
    EcephysElectrodeGroup,
    EcephysSpecimen,
    EcephysEyeTrackingRigMetadata,
    EcephysCSD)

class NwbHelper:

  DEFAULT_RUNNING_SPEED_UNITS = {
      "velocity": "cm/s",
      "vin": "V",
      "vsig": "V",
      "rotation": "radians"
  }

  # ParsedProbeData = Tuple[pd.DataFrame,  # unit_tables
  #                       Dict[int, np.ndarray],  # spike_times
  #                       Dict[int, np.ndarray],  # spike_amplitudes
  #                       Dict[int, np.ndarray]]  # mean_waveforms

  def __init__(self, session_data, probes):
    self._running_speed_path = session_data['running_speed_path']
    self._optotagging_table_path = session_data['optotagging_table_path']
    self._eye_tracking_rig_geometry = session_data['eye_tracking_rig_geometry']
    self._stimulus_table_path = session_data['stim_table_file']
    self._probes = probes

  def to_nwb(self, nwbfile: NWBFile):
    #running_speed, raw_running_data = read_running_speed(self._running_speed_path)
    # add_running_speed_to_nwbfile(nwbfile, running_speed)
    #add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)

    # running_speed, raw_running_data = self.read_running_speed(self._running_speed_path)
    # self.add_running_speed_to_nwbfile(nwbfile, running_speed)
    # self.add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)

    add_probewise_data_to_nwbfile(nwbfile, self._probes)

    if self._optotagging_table_path is not None:
      optotagging_table = pd.read_csv(self._optotagging_table_path)
      nwbfile = add_optotagging_table_to_nwbfile(nwbfile, optotagging_table)

    if self._eye_tracking_rig_geometry is not None:
      add_eye_tracking_rig_geometry_data_to_nwbfile(
          nwbfile,
          self._eye_tracking_rig_geometry
      )

    stimulus_columns_to_drop = [
        "colorSpace", "depth", "interpolate", "pos", "rgbPedestal", "tex",
        "texRes", "flipHoriz", "flipVert", "rgb", "signalDots"
    ]
    stimulus_table = \
        read_stimulus_table(self._stimulus_table_path,
                            columns_to_drop=stimulus_columns_to_drop)
    
    add_stimulus_timestamps(nwbfile, stimulus_table['start_time'].values)

    add_stimulus_presentations(nwbfile, stimulus_table)

  # def add_raw_running_data_to_nwbfile(self, nwbfile, raw_running_data, units=None):
  #   if units is None:
  #       units = NwbHelper.DEFAULT_RUNNING_SPEED_UNITS

  #   raw_rotation_timeseries = pynwb.base.TimeSeries(
  #       name="raw_running_wheel_rotation",
  #       timestamps=np.array(raw_running_data["frame_time"]),
  #       data=raw_running_data["dx"].values,
  #       unit=units["rotation"]
  #   )

  #   vsig_ts = pynwb.base.TimeSeries(
  #       name="running_wheel_signal_voltage",
  #       timestamps=raw_rotation_timeseries,
  #       data=raw_running_data["vsig"].values,
  #       unit=units["vsig"]
  #   )

  #   vin_ts = pynwb.base.TimeSeries(
  #       name="running_wheel_supply_voltage",
  #       timestamps=raw_rotation_timeseries,
  #       data=raw_running_data["vin"].values,
  #       unit=units["vin"]
  #   )

  #   nwbfile.add_acquisition(raw_rotation_timeseries)
  #   nwbfile.add_acquisition(vsig_ts)
  #   nwbfile.add_acquisition(vin_ts)

  #   return nwbfile

  def add_running_speed_to_nwbfile(self, nwbfile, running_speed, units=None):
    if units is None:
        units = NwbHelper.DEFAULT_RUNNING_SPEED_UNITS

    running_mod = pynwb.ProcessingModule("running", "running speed data")
    nwbfile.add_processing_module(running_mod)

    # print('running_speed', running_speed)

    running_speed_timeseries = pynwb.base.TimeSeries(
        name="running_speed",
        timestamps=running_speed["frame_time"].values,
        data=running_speed["velocity"].values,
        unit=units["velocity"]
    )

    # Create an 'empty' timeseries that only stores end times
    # # An array of nans needs to be created to avoid an nwb schema violation
    # running_speed_end_timeseries = pynwb.base.TimeSeries(
    #     name="running_speed_end_times",
    #     data=np.full(running_speed["velocity"].shape, np.nan),
    #     timestamps=running_speed["end_time"].values,
    #     unit=units["velocity"]
    # )

    rotation_timeseries = pynwb.base.TimeSeries(
        name="running_wheel_rotation",
        timestamps=running_speed_timeseries,
        data=running_speed["net_rotation"].values,
        unit=units["rotation"]
    )

    running_mod.add_data_interface(running_speed_timeseries)
    # running_mod.add_data_interface(running_speed_end_timeseries)
    running_mod.add_data_interface(rotation_timeseries)

    return nwbfile

  def read_running_speed(self, path):
    """ Reads running speed data and timestamps into a RunningSpeed named tuple

    Parameters
    ----------
    path : str
        path to running speed store


    Returns
    -------
    tuple :
        first item is dataframe of running speed data, second is dataframe of
        raw values (vsig, vin, encoder rotation)

    """

    return (
        pd.read_hdf(path, key="running_speed"),
        pd.read_hdf(path, key="raw_data")
    )

  def read_spike_times_to_dictionary(
      self, spike_times_path, spike_units_path, local_to_global_unit_map=None
  ):
    """ Reads spike times and assigned units from npy files into a lookup table.

    Parameters
    ----------
    spike_times_path : str
        npy file identifying, per spike, the time at which that spike occurred.
    spike_units_path : str
        npy file identifying, per spike, the unit associated with that spike.
        These are probe-local, so a local_to_global_unit_map is used to
        associate spikes with global unit identifiers.
    local_to_global_unit_map : dict, optional
        Maps probewise local unit indices to global unit ids

    Returns
    -------
    output_times : dict
        keys are unit identifiers, values are spike time arrays

    """

    spike_times = self.load_and_squeeze_npy(spike_times_path)
    spike_units = self.load_and_squeeze_npy(spike_units_path)

    return self.group_1d_by_unit(spike_times, spike_units, local_to_global_unit_map)

  def add_ecephys_electrodes(self, 
    nwbfile: pynwb.NWBFile,
    channels: List[dict],
    electrode_group: EcephysElectrodeGroup,
    local_index_whitelist: Optional[np.ndarray] = None
  ):
    """Add electrode information to an ecephys nwbfile electrode table.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The nwbfile to add electrodes data to
    channels : List[dict]
        A list of 'channel' dictionaries containing the following fields:
            id: The unique id for a given electrode/channel
            probe_id: The unique id for an electrode's/channel's device
            valid_data: Whether the data for an electrode/channel is usable
            local_index: The local index of an electrode/channel on a
                         given device
            probe_vertical_position: Length-wise position of electrode/channel
                         on device (microns)
            probe_horizontal_position: Width-wise position of electrode/channel
                         on device (microns)
            manual_structure_id: The LIMS id associated with an anatomical
                         structure
            manual_structure_acronym: Acronym associated with an anatomical
                         structure
            anterior_posterior_ccf_coordinate
            dorsal_ventral_ccf_coordinate
            left_right_ccf_coordinate

            Optional fields which may be used in the future:
            impedence: The impedence of a given channel.
            filtering: The type of hardware filtering done a channel.
                       (e.g. "1000 Hz low-pass filter")

    electrode_group : EcephysElectrodeGroup
        The pynwb electrode group that electrodes should be associated with
    local_index_whitelist : Optional[np.ndarray], optional
        If provided, only add electrodes (a.k.a. channels) specified by the
        whitelist (and in order specified), by default None
    """
    self.add_ecephys_electrode_columns(nwbfile)

    channel_table = pd.DataFrame(channels)

    if local_index_whitelist is not None:
        channel_table.set_index("local_index", inplace=True)
        channel_table = channel_table.loc[local_index_whitelist, :]
        channel_table.reset_index(inplace=True)

    for _, row in channel_table.iterrows():
        x = row["anterior_posterior_ccf_coordinate"]
        y = row["dorsal_ventral_ccf_coordinate"]
        z = row["left_right_ccf_coordinate"]

        nwbfile.add_electrode(
            id=row["id"],
            x=(np.nan if x is None else x),  # Not all probes have CCF coords
            y=(np.nan if y is None else y),
            z=(np.nan if z is None else z),
            probe_vertical_position=row["probe_vertical_position"],
            probe_horizontal_position=row["probe_horizontal_position"],
            local_index=row["local_index"],
            valid_data=row["valid_data"],
            probe_id=row["probe_id"],
            group=electrode_group,
            location=row["manual_structure_acronym"],
            imp=row["impedence"],
            filtering=row["filtering"]
        )

  def add_ecephys_electrodes(self,
    nwbfile: pynwb.NWBFile,
    channels: List[dict],
    electrode_group: EcephysElectrodeGroup,
    local_index_whitelist: Optional[np.ndarray] = None):
    """Add electrode information to an ecephys nwbfile electrode table.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The nwbfile to add electrodes data to
    channels : List[dict]
        A list of 'channel' dictionaries containing the following fields:
            id: The unique id for a given electrode/channel
            probe_id: The unique id for an electrode's/channel's device
            valid_data: Whether the data for an electrode/channel is usable
            local_index: The local index of an electrode/channel on a
                         given device
            probe_vertical_position: Length-wise position of electrode/channel
                         on device (microns)
            probe_horizontal_position: Width-wise position of electrode/channel
                         on device (microns)
            manual_structure_id: The LIMS id associated with an anatomical
                         structure
            manual_structure_acronym: Acronym associated with an anatomical
                         structure
            anterior_posterior_ccf_coordinate
            dorsal_ventral_ccf_coordinate
            left_right_ccf_coordinate

            Optional fields which may be used in the future:
            impedence: The impedence of a given channel.
            filtering: The type of hardware filtering done a channel.
                       (e.g. "1000 Hz low-pass filter")

    electrode_group : EcephysElectrodeGroup
        The pynwb electrode group that electrodes should be associated with
    local_index_whitelist : Optional[np.ndarray], optional
        If provided, only add electrodes (a.k.a. channels) specified by the
        whitelist (and in order specified), by default None
    """
    self.add_ecephys_electrode_columns(nwbfile)

    channel_table = pd.DataFrame(channels)

    if local_index_whitelist is not None:
        channel_table.set_index("local_index", inplace=True)
        channel_table = channel_table.loc[local_index_whitelist, :]
        channel_table.reset_index(inplace=True)

    for _, row in channel_table.iterrows():
        x = row["anterior_posterior_ccf_coordinate"]
        y = row["dorsal_ventral_ccf_coordinate"]
        z = row["left_right_ccf_coordinate"]

        nwbfile.add_electrode(
            id=row["id"],
            x=(np.nan if x is None else x),  # Not all probes have CCF coords
            y=(np.nan if y is None else y),
            z=(np.nan if z is None else z),
            probe_vertical_position=row["probe_vertical_position"],
            probe_horizontal_position=row["probe_horizontal_position"],
            local_index=row["local_index"],
            valid_data=row["valid_data"],
            probe_id=row["probe_id"],
            group=electrode_group,
            location=row["manual_structure_acronym"],
            imp=row["impedence"],
            filtering=row["filtering"]
        )

  # def add_probe_to_nwbfile(self, nwbfile, probe_id, sampling_rate, lfp_sampling_rate,
  #                        has_lfp_data, name,
  #                        location="See electrode locations"):
  #   """ Creates objects required for representation of a single
  #   extracellular ephys probe within an NWB file.

  #   Parameters
  #   ----------
  #   nwbfile : pynwb.NWBFile
  #       file to which probe information will be assigned.
  #   probe_id : int
  #       unique identifier for this probe
  #   sampling_rate: float,
  #       sampling rate of the neuropixels probe
  #   lfp_sampling_rate: float
  #       sampling rate of LFP
  #   has_lfp_data: bool
  #       True if LFP data is available for the probe, otherwise False
  #   name : str, optional
  #       human-readable name for this probe.
  #       Practically, we use tags like "probeA" or "probeB"
  #   location : str, optional
  #       A required field for the `EcephysElectrodeGroup`. Because the group
  #       contains a number of electrodes/channels along the neuropixels probe,
  #       location will vary significantly. Thus by default this field is:
  #       "See electrode locations" where the nwbfile.electrodes table will
  #       provide much more detailed location information.

  #   Returns
  #   ------
  #       nwbfile : pynwb.NWBFile
  #           the updated file object
  #       probe_nwb_device : pynwb.device.Device
  #           device object corresponding to this probe
  #       probe_nwb_electrode_group : pynwb.ecephys.ElectrodeGroup
  #           electrode group object corresponding to this probe

  #   """
  #   probe_nwb_device = EcephysProbe(name=name,
  #                                   description="Neuropixels 1.0 Probe",
  #                                   manufacturer="imec",
  #                                   probe_id=probe_id,
  #                                   sampling_rate=sampling_rate)

  #   probe_nwb_electrode_group = EcephysElectrodeGroup(
  #       name=name,
  #       description="Ecephys Electrode Group",  # required field
  #       probe_id=probe_id,
  #       location=location,
  #       device=probe_nwb_device,
  #       lfp_sampling_rate=lfp_sampling_rate,
  #       has_lfp_data=has_lfp_data
  #   )

  #   nwbfile.add_device(probe_nwb_device)
  #   nwbfile.add_electrode_group(probe_nwb_electrode_group)

  #   return nwbfile, probe_nwb_device, probe_nwb_electrode_group

  # def parse_probes_data(self, probes: List[Dict[str, Any]]) -> ParsedProbeData:
  #   """Given a list of probe dictionaries specifying data file locations, load
  #   and parse probe data into intermediate data structures needed for adding
  #   probe data to an nwbfile.

  #   Parameters
  #   ----------
  #   probes : List[Dict[str, Any]]
  #       A list of dictionaries (one entry for each probe), where each probe
  #       dictionary contains metadata (id, name, sampling_rate, etc...) as well
  #       as filepaths pointing to where probe lfp data can be found.

  #   Returns
  #   -------
  #   ParsedProbeData : Tuple[...]
  #       unit_tables : pd.DataFrame
  #           A table containing unit metadata from all probes.
  #       spike_times : Dict[int, np.ndarray]
  #           Keys: unit identifiers, Values: spike time arrays
  #       spike_amplitudes : Dict[int, np.ndarray]
  #           Keys: unit identifiers, Values: spike amplitude arrays
  #       mean_waveforms : Dict[int, np.ndarray]
  #           Keys: unit identifiers, Values: mean waveform arrays
  #   """

  #   unit_tables = []
  #   spike_times = {}
  #   spike_amplitudes = {}
  #   mean_waveforms = {}

  #   for probe in probes:
  #       unit_tables.append(pd.DataFrame(probe['units']))

  #       local_to_global_unit_map = \
  #           {unit['cluster_id']: unit['id'] for unit in probe['units']}

  #       spike_times.update(self.read_spike_times_to_dictionary(
  #           probe['spike_times_path'],
  #           probe['spike_clusters_file'],
  #           local_to_global_unit_map
  #       ))
  #       mean_waveforms.update(self.read_waveforms_to_dictionary(
  #           probe['mean_waveforms_path'],
  #           local_to_global_unit_map
  #       ))

  #       spike_amplitudes.update(self.read_spike_amplitudes_to_dictionary(
  #           probe["spike_amplitudes_path"],
  #           probe["spike_clusters_file"],
  #           probe["templates_path"],
  #           probe["spike_templates_path"],
  #           probe["inverse_whitening_matrix_path"],
  #           local_to_global_unit_map=local_to_global_unit_map,
  #           scale_factor=probe["amplitude_scale_factor"]
  #       ))

  #   units_table = pd.concat(unit_tables).set_index(keys='id', drop=True)

  #   return (units_table, spike_times, spike_amplitudes, mean_waveforms)

  # def read_waveforms_to_dictionary(
  #   self, waveforms_path, local_to_global_unit_map=None, peak_channel_map=None
  # ):
  #   """ Builds a lookup table for unitwise waveform data

  #   Parameters
  #   ----------
  #   waveforms_path : str
  #       npy file containing waveform data for each unit. Dimensions ought to
  #       be units X samples X channels
  #   local_to_global_unit_map : dict, optional
  #       Maps probewise local unit indices to global unit ids
  #   peak_channel_map : dict, optional
  #       Maps unit identifiers to indices of peak channels. If provided,
  #       the output will contain only samples on the peak
  #       channel for each unit.

  #   Returns
  #   -------
  #   output_waveforms : dict
  #       Keys are unit identifiers, values are samples X channels data arrays.

  #   """

  #   waveforms = np.squeeze(np.load(waveforms_path, allow_pickle=False))
  #   output_waveforms = {}
  #   for unit_id, waveform in enumerate(
  #       np.split(waveforms, waveforms.shape[0], axis=0)
  #   ):
  #       if local_to_global_unit_map is not None:
  #           if unit_id not in local_to_global_unit_map:
  #               # logging.warning(
  #               #     f"""unable to find unit at local position
  #               #         {unit_id} while reading waveforms"""
  #               # )
  #               continue
  #           unit_id = local_to_global_unit_map[unit_id]

  #       if peak_channel_map is not None:
  #           waveform = waveform[:, peak_channel_map[unit_id]]

  #       output_waveforms[unit_id] = np.squeeze(waveform)

  #   return output_waveforms


  # def load_and_squeeze_npy(self, path):
  #     return np.squeeze(np.load(path, allow_pickle=False))


  # def fill_df(self, df, str_fill=""):
  #     df = df.copy()

  #     for colname in df.columns:
  #         if not pd.api.types.is_numeric_dtype(df[colname]):
  #             df[colname].fillna(str_fill)

  #         if np.all(pd.isna(df[colname]).values):
  #             df[colname] = [str_fill for ii in range(df.shape[0])]

  #         if pd.api.types.is_string_dtype(df[colname]):
  #             df[colname] = df[colname].astype(str)

  #     return df

  # def read_spike_amplitudes_to_dictionary(
  #   self,
  #   spike_amplitudes_path, spike_units_path,
  #   templates_path, spike_templates_path, inverse_whitening_matrix_path,
  #   local_to_global_unit_map=None,
  #   scale_factor=1.0
  # ):

  #   spike_amplitudes = self.load_and_squeeze_npy(spike_amplitudes_path)
  #   spike_units = self.load_and_squeeze_npy(spike_units_path)

  #   templates = self.load_and_squeeze_npy(templates_path)
  #   spike_templates = self.load_and_squeeze_npy(spike_templates_path)
  #   inverse_whitening_matrix = \
  #       self.load_and_squeeze_npy(inverse_whitening_matrix_path)

  #   for temp_idx in range(templates.shape[0]):
  #       templates[temp_idx, :, :] = np.dot(
  #           np.ascontiguousarray(templates[temp_idx, :, :]),
  #           np.ascontiguousarray(inverse_whitening_matrix)
  #       )

  #   scaled_amplitudes = self.scale_amplitudes(spike_amplitudes,
  #                                        templates,
  #                                        spike_templates,
  #                                        scale_factor=scale_factor)

  #   return self.group_1d_by_unit(scaled_amplitudes,
  #                           spike_units,
  #                           local_to_global_unit_map)

  # def scale_amplitudes(self,
  #   spike_amplitudes,
  #   templates,
  #   spike_templates,
  #   scale_factor=1.0
  # ):

  #   template_full_amplitudes = templates.max(axis=1) - templates.min(axis=1)
  #   template_amplitudes = template_full_amplitudes.max(axis=1)

  #   template_amplitudes = template_amplitudes[spike_templates]
  #   spike_amplitudes = template_amplitudes * spike_amplitudes * scale_factor

  #   return spike_amplitudes


  # def read_spike_times_to_dictionary(
  #     self,
  #     spike_times_path,
  #     spike_units_path,
  #     local_to_global_unit_map=None
  # ):
  #     """ Reads spike times and assigned units from npy files into a lookup table.

  #     Parameters
  #     ----------
  #     spike_times_path : str
  #         npy file identifying, per spike, the time at which that spike occurred.
  #     spike_units_path : str
  #         npy file identifying, per spike, the unit associated with that spike.
  #         These are probe-local, so a local_to_global_unit_map is used to
  #         associate spikes with global unit identifiers.
  #     local_to_global_unit_map : dict, optional
  #         Maps probewise local unit indices to global unit ids

  #     Returns
  #     -------
  #     output_times : dict
  #         keys are unit identifiers, values are spike time arrays

  #     """

  #     spike_times = self.load_and_squeeze_npy(spike_times_path)
  #     spike_units = self.load_and_squeeze_npy(spike_units_path)

  #     return self.group_1d_by_unit(spike_times, spike_units, local_to_global_unit_map)

  # def group_1d_by_unit(self, data, data_unit_map, local_to_global_unit_map=None):
  #   sort_order = np.argsort(data_unit_map, kind="stable")
  #   data_unit_map = data_unit_map[sort_order]
  #   data = data[sort_order]

  #   changes = np.concatenate(
  #       [
  #           np.array([0]),
  #           np.where(np.diff(data_unit_map))[0] + 1,
  #           np.array([data.size]),
  #       ]
  #   )

  #   output = {}
  #   for jj, (low, high) in enumerate(zip(changes[:-1], changes[1:])):
  #       local_unit = data_unit_map[low]
  #       current = data[low:high]

  #       if local_to_global_unit_map is not None:
  #           if local_unit not in local_to_global_unit_map:
  #               # logging.warning(
  #               #     f"unable to find unit at local position {local_unit}"
  #               # )
  #               continue
  #           global_id = local_to_global_unit_map[local_unit]
  #           output[global_id] = current
  #       else:
  #           output[local_unit] = current

  #   return output

  # def add_probewise_data_to_nwbfile(self, nwbfile, probes):
  #   """ Adds channel (electrode) and spike data for a single probe to
  #       the session-level nwb file.
  #   """
  #   for probe in probes:

  #       if probe.get("temporal_subsampling_factor", None) is not None:
  #           probe["lfp_sampling_rate"] = probe["lfp_sampling_rate"] / \
  #               probe["temporal_subsampling_factor"]

  #       nwbfile, probe_nwb_device, probe_nwb_electrode_group = \
  #           self.add_probe_to_nwbfile(
  #               nwbfile,
  #               probe_id=probe["id"],
  #               name=probe["name"],
  #               sampling_rate=probe["sampling_rate"],
  #               lfp_sampling_rate=probe["lfp_sampling_rate"],
  #               has_lfp_data=probe["lfp"] is not None
  #           )

  #       self.add_ecephys_electrodes(nwbfile,
  #                              probe["channels"],
  #                              probe_nwb_electrode_group)

  #   units_table, spike_times, spike_amplitudes, mean_waveforms = \
  #       self.parse_probes_data(probes)
  #   nwbfile.units = pynwb.misc.Units.from_dataframe(self.fill_df(units_table),
  #                                                   name='units')

  #   sorted_spike_times, sorted_spike_amplitudes = \
  #       self.filter_and_sort_spikes(spike_times, spike_amplitudes)

  #   self.add_ragged_data_to_dynamic_table(
  #       table=nwbfile.units,
  #       data=sorted_spike_times,
  #       column_name="spike_times",
  #       column_description="times (s) of detected spiking events",
  #   )

  #   self.add_ragged_data_to_dynamic_table(
  #       table=nwbfile.units,
  #       data=sorted_spike_amplitudes,
  #       column_name="spike_amplitudes",
  #       column_description="amplitude (s) of detected spiking events"
  #   )

  #   self.add_ragged_data_to_dynamic_table(
  #       table=nwbfile.units,
  #       data=mean_waveforms,
  #       column_name="waveform_mean",
  #       column_description="mean waveforms on peak channels (over samples)",
  #   )

  #   return nwbfile

  # def add_ragged_data_to_dynamic_table(
  #   self, table, data, column_name, column_description=""
  # ):
  #     """ Builds the index and data vectors required for writing ragged array
  #     data to a pynwb dynamic table

  #     Parameters
  #     ----------
  #     table : pynwb.core.DynamicTable
  #         table to which data will be added (as VectorData / VectorIndex)
  #     data : dict
  #         each key-value pair describes some grouping of data
  #     column_name : str
  #         used to set the name of this column
  #     column_description : str, optional
  #         used to set the description of this column

  #     Returns
  #     -------
  #     nwbfile : pynwb.NWBFile

  #     """

  #     idx, values = dict_to_indexed_array(data, table.id.data)
  #     del data

  #     table.add_column(
  #         name=column_name,
  #         description=column_description,
  #         data=values,
  #         index=idx
  #     )


  # DEFAULT_RUNNING_SPEED_UNITS = {
  #     "velocity": "cm/s",
  #     "vin": "V",
  #     "vsig": "V",
  #     "rotation": "radians"
  # }

  # def filter_and_sort_spikes(
  #   self, 
  #   spike_times_mapping:
  #   Dict[int, np.ndarray],
  #   spike_amplitudes_mapping:
  #   Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray],
  #                                  Dict[int, np.ndarray]]:
  #   """Filter out invalid spike timepoints and sort spike data
  #   (times + amplitudes) by times.

  #   Parameters
  #   ----------
  #   spike_times_mapping : Dict[int, np.ndarray]
  #       Keys: unit identifiers, Values: spike time arrays
  #   spike_amplitudes_mapping : Dict[int, np.ndarray]
  #       Keys: unit identifiers, Values: spike amplitude arrays

  #   Returnsread_waveforms_to_dictionary
  #   -------
  #   Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]
  #       A tuple containing filtered and sorted spike_times_mapping and
  #       spike_amplitudes_mapping data.
  #   """
  #   sorted_spike_times_mapping = {}
  #   sorted_spike_amplitudes_mapping = {}

  #   for unit_id, _ in spike_times_mapping.items():
  #       spike_times = spike_times_mapping[unit_id]
  #       spike_amplitudes = spike_amplitudes_mapping[unit_id]

  #       valid = spike_times >= 0
  #       filtered_spike_times = spike_times[valid]
  #       filtered_spike_amplitudes = spike_amplitudes[valid]

  #       order = np.argsort(filtered_spike_times)
  #       sorted_spike_times = filtered_spike_times[order]
  #       sorted_spike_amplitudes = filtered_spike_amplitudes[order]

  #       sorted_spike_times_mapping[unit_id] = sorted_spike_times
  #       sorted_spike_amplitudes_mapping[unit_id] = sorted_spike_amplitudes

  #   return (sorted_spike_times_mapping, sorted_spike_amplitudes_mapping)

  # def add_ecephys_electrode_columns(self, nwbfile: pynwb.NWBFile,
  #                                 columns_to_add:
  #                                 Optional[List[Tuple[str, str]]] = None):
  #   """Add additional columns to ecephys nwbfile electrode table.

  #   Parameters
  #   ----------
  #   nwbfile : pynwb.NWBFile
  #       An nwbfile to add additional electrode columns to
  #   columns_to_add : Optional[List[Tuple[str, str]]]
  #       A list of (column_name, column_description) tuples to be added
  #       to the nwbfile electrode table, by default None. If None, default
  #       columns are added.
  #   """
  #   default_columns = [
  #       ("probe_vertical_position",
  #        "Length-wise position of electrode/channel on device (microns)"),
  #       ("probe_horizontal_position",
  #        "Width-wise position of electrode/channel on device (microns)"),
  #       ("probe_id", "The unique id of this electrode's/channel's device"),
  #       ("local_index", "The local index of electrode/channel on device"),
  #       ("valid_data", "Whether data from this electrode/channel is usable")
  #   ]

  #   if columns_to_add is None:
  #       columns_to_add = default_columns

  #   for col_name, col_description in columns_to_add:
  #       if (not nwbfile.electrodes) or \
  #               (col_name not in nwbfile.electrodes.colnames):
  #           nwbfile.add_electrode_column(name=col_name,
  #                                        description=col_description)
