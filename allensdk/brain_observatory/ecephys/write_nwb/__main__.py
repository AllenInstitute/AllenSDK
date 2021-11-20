import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path, PurePath
import multiprocessing as mp
from functools import partial

import h5py
import pynwb
import requests
import pandas as pd
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO

from allensdk.config.manifest import Manifest

from ._schemas import InputSchema, OutputSchema
from allensdk.brain_observatory.nwb import setup_table_for_invalid_times  # noqa: F401
from allensdk.brain_observatory.nwb import (
    add_stimulus_presentations,
    add_stimulus_timestamps,
    add_invalid_times,
    setup_table_for_epochs,
    read_eye_dlc_tracking_ellipses,
    read_eye_gaze_mappings,
    add_eye_tracking_ellipse_fit_data_to_nwbfile,
    add_eye_gaze_mapping_data_to_nwbfile,
    eye_tracking_data_is_valid
)
from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs, optional_lims_inputs
)
from allensdk.brain_observatory import dict_to_indexed_array
from allensdk.brain_observatory.ecephys.file_io.continuous_file import ContinuousFile
from allensdk.brain_observatory.ecephys.nwb import (EcephysProbe,
                                                    EcephysElectrodeGroup,
                                                    EcephysSpecimen,
                                                    EcephysEyeTrackingRigMetadata,
                                                    EcephysCSD)
from allensdk.brain_observatory.sync_dataset import Dataset
import allensdk.brain_observatory.sync_utilities as su


STIM_TABLE_RENAMES_MAP = {"Start": "start_time", "End": "stop_time"}


def load_and_squeeze_npy(path):
    return np.squeeze(np.load(path, allow_pickle=False))


def fill_df(df, str_fill=""):
    df = df.copy()

    for colname in df.columns:
        if not pd.api.types.is_numeric_dtype(df[colname]):
            df[colname].fillna(str_fill)

        if np.all(pd.isna(df[colname]).values):
            df[colname] = [str_fill for ii in range(df.shape[0])]

        if pd.api.types.is_string_dtype(df[colname]):
            df[colname] = df[colname].astype(str)

    return df


def get_inputs_from_lims(host, ecephys_session_id, output_root, job_queue, strategy):
    """
     This is a development / testing utility for running this module from the Allen Institute for Brain Science's
    Laboratory Information Management System (LIMS). It will only work if you are on our internal network.

    Parameters
    ----------
    ecephys_session_id : int
        Unique identifier for session of interest.
    output_root : str
        Output file will be written into this directory.
    job_queue : str
        Identifies the job queue from which to obtain configuration data
    strategy : str
        Identifies the LIMS strategy which will be used to write module inputs.

    Returns
    -------
    data : dict
        Response from LIMS. Should meet the schema defined in _schemas.py

    """

    uri = f"{host}/input_jsons?object_id={ecephys_session_id}&object_class=EcephysSession&strategy_class={strategy}&job_queue_name={job_queue}&output_directory={output_root}"
    response = requests.get(uri)
    data = response.json()

    if len(data) == 1 and "error" in data:
        raise ValueError("bad request uri: {} ({})".format(uri, data["error"]))

    return data


def read_stimulus_table(path: str,
                        column_renames_map: Dict[str, str] = None,
                        columns_to_drop: List[str] = None) -> pd.DataFrame:
    """ Loads from a CSV on disk the stimulus table for this session.
    Optionally renames columns to match NWB epoch specifications.

    Parameters
    ----------
    path : str
        path to stimulus table csv
    column_renames_map : Dict[str, str], optional
        If provided, will be used to rename columns from keys -> values.
        Default renames: ('Start' -> 'start_time') and ('End' -> 'stop_time')
    columns_to_drop : List, optional
        A list of column names to drop. Columns will be dropped BEFORE
        any renaming occurs. If None, no columns are dropped.
        By default None.

    Returns
    -------
    pd.DataFrame :
        stimulus table with applied renames

    """
    if column_renames_map is None:
        column_renames_map = STIM_TABLE_RENAMES_MAP

    ext = PurePath(path).suffix

    if ext == ".csv":
        stimulus_table = pd.read_csv(path)
    else:
        raise IOError(f"unrecognized stimulus table extension: {ext}")

    if columns_to_drop:
        stimulus_table = stimulus_table.drop(errors='ignore',
                                             columns=columns_to_drop)

    return stimulus_table.rename(columns=column_renames_map, index={})


def read_spike_times_to_dictionary(
    spike_times_path, spike_units_path, local_to_global_unit_map=None
):
    """ Reads spike times and assigned units from npy files into a lookup table.

    Parameters
    ----------
    spike_times_path : str
        npy file identifying, per spike, the time at which that spike occurred.
    spike_units_path : str
        npy file identifying, per spike, the unit associated with that spike. These are probe-local, so a
        local_to_global_unit_map is used to associate spikes with global unit identifiers.
    local_to_global_unit_map : dict, optional
        Maps probewise local unit indices to global unit ids

    Returns
    -------
    output_times : dict
        keys are unit identifiers, values are spike time arrays

    """

    spike_times = load_and_squeeze_npy(spike_times_path)
    spike_units = load_and_squeeze_npy(spike_units_path)

    return group_1d_by_unit(spike_times, spike_units, local_to_global_unit_map)


def read_spike_amplitudes_to_dictionary(
    spike_amplitudes_path, spike_units_path,
    templates_path, spike_templates_path, inverse_whitening_matrix_path,
    local_to_global_unit_map=None,
    scale_factor=1.0
):

    spike_amplitudes = load_and_squeeze_npy(spike_amplitudes_path)
    spike_units = load_and_squeeze_npy(spike_units_path)

    templates = load_and_squeeze_npy(templates_path)
    spike_templates = load_and_squeeze_npy(spike_templates_path)
    inverse_whitening_matrix = load_and_squeeze_npy(inverse_whitening_matrix_path)

    for temp_idx in range(templates.shape[0]):
        templates[temp_idx, :, :] = np.dot(
            np.ascontiguousarray(templates[temp_idx, :, :]),
            np.ascontiguousarray(inverse_whitening_matrix)
        )

    scaled_amplitudes = scale_amplitudes(spike_amplitudes, templates, spike_templates, scale_factor=scale_factor)
    return group_1d_by_unit(scaled_amplitudes, spike_units, local_to_global_unit_map)


def scale_amplitudes(spike_amplitudes, templates, spike_templates, scale_factor=1.0):

    template_full_amplitudes = templates.max(axis=1) - templates.min(axis=1)
    template_amplitudes = template_full_amplitudes.max(axis=1)

    template_amplitudes = template_amplitudes[spike_templates]
    spike_amplitudes = template_amplitudes * spike_amplitudes * scale_factor
    return spike_amplitudes


def filter_and_sort_spikes(spike_times_mapping: Dict[int, np.ndarray],
                           spike_amplitudes_mapping: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Filter out invalid spike timepoints and sort spike data
    (times + amplitudes) by times.

    Parameters
    ----------
    spike_times_mapping : Dict[int, np.ndarray]
        Keys: unit identifiers, Values: spike time arrays
    spike_amplitudes_mapping : Dict[int, np.ndarray]
        Keys: unit identifiers, Values: spike amplitude arrays

    Returns
    -------
    Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]
        A tuple containing filtered and sorted spike_times_mapping and
        spike_amplitudes_mapping data.
    """
    sorted_spike_times_mapping = {}
    sorted_spike_amplitudes_mapping = {}

    for unit_id, _ in spike_times_mapping.items():
        spike_times = spike_times_mapping[unit_id]
        spike_amplitudes = spike_amplitudes_mapping[unit_id]

        valid = spike_times >= 0
        filtered_spike_times = spike_times[valid]
        filtered_spike_amplitudes = spike_amplitudes[valid]

        order = np.argsort(filtered_spike_times)
        sorted_spike_times = filtered_spike_times[order]
        sorted_spike_amplitudes = filtered_spike_amplitudes[order]

        sorted_spike_times_mapping[unit_id] = sorted_spike_times
        sorted_spike_amplitudes_mapping[unit_id] = sorted_spike_amplitudes

    return (sorted_spike_times_mapping, sorted_spike_amplitudes_mapping)


def group_1d_by_unit(data, data_unit_map, local_to_global_unit_map=None):
    sort_order = np.argsort(data_unit_map, kind="stable")
    data_unit_map = data_unit_map[sort_order]
    data = data[sort_order]

    changes = np.concatenate(
        [
            np.array([0]),
            np.where(np.diff(data_unit_map))[0] + 1,
            np.array([data.size]),
        ]
    )

    output = {}
    for jj, (low, high) in enumerate(zip(changes[:-1], changes[1:])):
        local_unit = data_unit_map[low]
        current = data[low:high]

        if local_to_global_unit_map is not None:
            if local_unit not in local_to_global_unit_map:
                logging.warning(
                    f"unable to find unit at local position {local_unit}"
                )
                continue
            global_id = local_to_global_unit_map[local_unit]
            output[global_id] = current
        else:
            output[local_unit] = current

    return output


def add_metadata_to_nwbfile(nwbfile, input_metadata):
    metadata = input_metadata.copy()

    if "full_genotype" in metadata:
        metadata["genotype"] = metadata.pop("full_genotype")

    if "stimulus_name" in metadata:
        nwbfile.stimulus_notes = metadata.pop("stimulus_name")

    if "age_in_days" in metadata:
        metadata["age"] = f"P{int(metadata['age_in_days'])}D"

    if "donor_id" in metadata:
        metadata["subject_id"] = str(metadata.pop("donor_id"))

    nwbfile.subject = EcephysSpecimen(**metadata)
    return nwbfile


def read_waveforms_to_dictionary(
    waveforms_path, local_to_global_unit_map=None, peak_channel_map=None
):
    """ Builds a lookup table for unitwise waveform data

    Parameters
    ----------
    waveforms_path : str
        npy file containing waveform data for each unit. Dimensions ought to be units X samples X channels
    local_to_global_unit_map : dict, optional
        Maps probewise local unit indices to global unit ids
    peak_channel_map : dict, optional
        Maps unit identifiers to indices of peak channels. If provided, the output will contain only samples on the peak
        channel for each unit.

    Returns
    -------
    output_waveforms : dict
        Keys are unit identifiers, values are samples X channels data arrays.

    """

    waveforms = np.squeeze(np.load(waveforms_path, allow_pickle=False))
    output_waveforms = {}
    for unit_id, waveform in enumerate(
        np.split(waveforms, waveforms.shape[0], axis=0)
    ):
        if local_to_global_unit_map is not None:
            if unit_id not in local_to_global_unit_map:
                logging.warning(
                    f"unable to find unit at local position {unit_id} while reading waveforms"
                )
                continue
            unit_id = local_to_global_unit_map[unit_id]

        if peak_channel_map is not None:
            waveform = waveform[:, peak_channel_map[unit_id]]

        output_waveforms[unit_id] = np.squeeze(waveform)

    return output_waveforms


def read_running_speed(path):
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


def add_probe_to_nwbfile(nwbfile, probe_id, sampling_rate, lfp_sampling_rate,
                         has_lfp_data, name,
                         location="See electrode locations"):
    """ Creates objects required for representation of a single
    extracellular ephys probe within an NWB file.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        file to which probe information will be assigned.
    probe_id : int
        unique identifier for this probe
    sampling_rate: float,
        sampling rate of the neuropixels probe
    lfp_sampling_rate: float
        sampling rate of LFP
    has_lfp_data: bool
        True if LFP data is available for the probe, otherwise False
    name : str, optional
        human-readable name for this probe.
        Practically, we use tags like "probeA" or "probeB"
    location : str, optional
        A required field for the `EcephysElectrodeGroup`. Because the group
        contains a number of electrodes/channels along the neuropixels probe,
        location will vary significantly. Thus by default this field is:
        "See electrode locations" where the nwbfile.electrodes table will
        provide much more detailed location information.

    Returns
    ------
        nwbfile : pynwb.NWBFile
            the updated file object
        probe_nwb_device : pynwb.device.Device
            device object corresponding to this probe
        probe_nwb_electrode_group : pynwb.ecephys.ElectrodeGroup
            electrode group object corresponding to this probe

    """
    probe_nwb_device = EcephysProbe(name=name,
                                    description="Neuropixels 1.0 Probe",  # required field
                                    manufacturer="imec",
                                    probe_id=probe_id,
                                    sampling_rate=sampling_rate)

    probe_nwb_electrode_group = EcephysElectrodeGroup(
        name=name,
        description="Ecephys Electrode Group",  # required field
        probe_id=probe_id,
        location=location,
        device=probe_nwb_device,
        lfp_sampling_rate=lfp_sampling_rate,
        has_lfp_data=has_lfp_data
    )

    nwbfile.add_device(probe_nwb_device)
    nwbfile.add_electrode_group(probe_nwb_electrode_group)

    return nwbfile, probe_nwb_device, probe_nwb_electrode_group


def add_ecephys_electrode_columns(nwbfile: pynwb.NWBFile,
                                  columns_to_add: Optional[List[Tuple[str, str]]] = None):
    """Add additional columns to ecephys nwbfile electrode table.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        An nwbfile to add additional electrode columns to
    columns_to_add : Optional[List[Tuple[str, str]]]
        A list of (column_name, column_description) tuples to be added
        to the nwbfile electrode table, by default None. If None, default
        columns are added.
    """
    default_columns = [
        ("probe_vertical_position", "Length-wise position of electrode/channel on device (microns)"),
        ("probe_horizontal_position", "Width-wise position of electrode/channel on device (microns)"),
        ("probe_id", "The unique id of this electrode's/channel's device"),
        ("local_index", "The local index of electrode/channel on device"),
        ("valid_data", "Whether data from this electrode/channel is usable")
    ]

    if columns_to_add is None:
        columns_to_add = default_columns

    for col_name, col_description in columns_to_add:
        if (not nwbfile.electrodes) or (col_name not in nwbfile.electrodes.colnames):
            nwbfile.add_electrode_column(name=col_name,
                                         description=col_description)


def add_ecephys_electrodes(nwbfile: pynwb.NWBFile,
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
            local_index: The local index of an electrode/channel on a given device
            probe_vertical_position: Length-wise position of electrode/channel on device (microns)
            probe_horizontal_position: Width-wise position of electrode/channel on device (microns)
            manual_structure_id: The LIMS id associated with an anatomical structure
            manual_structure_acronym: Acronym associated with an anatomical structure
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
    add_ecephys_electrode_columns(nwbfile)

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


def add_ragged_data_to_dynamic_table(
    table, data, column_name, column_description=""
):
    """ Builds the index and data vectors required for writing ragged array data to a pynwb dynamic table

    Parameters
    ----------
    table : pynwb.core.DynamicTable
        table to which data will be added (as VectorData / VectorIndex)
    data : dict
        each key-value pair describes some grouping of data
    column_name : str
        used to set the name of this column
    column_description : str, optional
        used to set the description of this column

    Returns
    -------
    nwbfile : pynwb.NWBFile

    """

    idx, values = dict_to_indexed_array(data, table.id.data)
    del data

    table.add_column(
        name=column_name, description=column_description, data=values, index=idx
    )


DEFAULT_RUNNING_SPEED_UNITS = {
    "velocity": "cm/s",
    "vin": "V",
    "vsig": "V",
    "rotation": "radians"
}


def add_running_speed_to_nwbfile(nwbfile, running_speed, units=None):
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    running_mod = pynwb.ProcessingModule("running", "running speed data")
    nwbfile.add_processing_module(running_mod)

    running_speed_timeseries = pynwb.base.TimeSeries(
        name="running_speed",
        timestamps=running_speed["start_time"].values,
        data=running_speed["velocity"].values,
        unit=units["velocity"]
    )

    # Create an 'empty' timeseries that only stores end times
    # An array of nans needs to be created to avoid an nwb schema violation
    running_speed_end_timeseries = pynwb.base.TimeSeries(
        name="running_speed_end_times",
        data=np.full(running_speed["velocity"].shape, np.nan),
        timestamps=running_speed["end_time"].values,
        unit=units["velocity"]
    )

    rotation_timeseries = pynwb.base.TimeSeries(
        name="running_wheel_rotation",
        timestamps=running_speed_timeseries,
        data=running_speed["net_rotation"].values,
        unit=units["rotation"]
    )

    running_mod.add_data_interface(running_speed_timeseries)
    running_mod.add_data_interface(running_speed_end_timeseries)
    running_mod.add_data_interface(rotation_timeseries)

    return nwbfile


def add_raw_running_data_to_nwbfile(nwbfile, raw_running_data, units=None):
    if units is None:
        units = DEFAULT_RUNNING_SPEED_UNITS

    raw_rotation_timeseries = pynwb.base.TimeSeries(
        name="raw_running_wheel_rotation",
        timestamps=np.array(raw_running_data["frame_time"]),
        data=raw_running_data["dx"].values,
        unit=units["rotation"]
    )

    vsig_ts = pynwb.base.TimeSeries(
        name="running_wheel_signal_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vsig"].values,
        unit=units["vsig"]
    )

    vin_ts = pynwb.base.TimeSeries(
        name="running_wheel_supply_voltage",
        timestamps=raw_rotation_timeseries,
        data=raw_running_data["vin"].values,
        unit=units["vin"]
    )

    nwbfile.add_acquisition(raw_rotation_timeseries)
    nwbfile.add_acquisition(vsig_ts)
    nwbfile.add_acquisition(vin_ts)

    return nwbfile


def write_probe_lfp_file(session_id, session_metadata, session_start_time,
                         log_level, probe):
    """ Writes LFP data (and associated channel information) for one
    probe to a standalone nwb file
    """

    logging.getLogger('').setLevel(log_level)
    logging.info(f"writing lfp file for probe {probe['id']}")

    nwbfile = pynwb.NWBFile(
        session_description='LFP data and associated channel info for a single Ecephys probe',
        identifier=f"{probe['id']}",
        session_id=f"{session_id}",
        session_start_time=session_start_time,
        institution="Allen Institute for Brain Science"
    )

    if session_metadata is not None:
        nwbfile = add_metadata_to_nwbfile(nwbfile, session_metadata)

    if probe.get("temporal_subsampling_factor", None) is not None:
        probe["lfp_sampling_rate"] = probe["lfp_sampling_rate"] / probe["temporal_subsampling_factor"]

    nwbfile, probe_nwb_device, probe_nwb_electrode_group = add_probe_to_nwbfile(
        nwbfile,
        probe_id=probe["id"],
        name=probe["name"],
        sampling_rate=probe["sampling_rate"],
        lfp_sampling_rate=probe["lfp_sampling_rate"],
        has_lfp_data=probe["lfp"] is not None
    )

    lfp_channels = np.load(probe['lfp']['input_channels_path'],
                           allow_pickle=False)

    add_ecephys_electrodes(nwbfile, probe["channels"],
                           probe_nwb_electrode_group,
                           local_index_whitelist=lfp_channels)

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=np.arange(len(nwbfile.electrodes)).tolist(),  # must use raw indices here
        name='electrodes',
        description=f"lfp channels on probe {probe['id']}"
    )

    lfp_data, lfp_timestamps = ContinuousFile(
        data_path=probe['lfp']['input_data_path'],
        timestamps_path=probe['lfp']['input_timestamps_path'],
        total_num_channels=len(nwbfile.electrodes)
    ).load(memmap=False)

    lfp_data = lfp_data.astype(np.float32)
    lfp_data = lfp_data * probe["amplitude_scale_factor"]

    lfp = pynwb.ecephys.LFP(name=f"probe_{probe['id']}_lfp")

    nwbfile.add_acquisition(lfp.create_electrical_series(
        name=f"probe_{probe['id']}_lfp_data",
        data=H5DataIO(data=lfp_data, compression='gzip', compression_opts=9),
        timestamps=H5DataIO(data=lfp_timestamps, compression='gzip', compression_opts=9),
        electrodes=electrode_table_region
    ))

    nwbfile.add_acquisition(lfp)

    csd, csd_times, csd_locs = read_csd_data_from_h5(probe["csd_path"])
    nwbfile = add_csd_to_nwbfile(nwbfile, csd, csd_times, csd_locs)

    with pynwb.NWBHDF5IO(probe['lfp']['output_path'], 'w') as lfp_writer:
        logging.info(f"writing probe lfp file to {probe['lfp']['output_path']}")
        lfp_writer.write(nwbfile, cache_spec=True)
    return {"id": probe["id"], "nwb_path": probe["lfp"]["output_path"]}


def read_csd_data_from_h5(csd_path):
    with h5py.File(csd_path, "r") as csd_file:
        return (csd_file["current_source_density"][:],
                csd_file["timestamps"][:],
                csd_file["csd_locations"][:])


def add_csd_to_nwbfile(nwbfile: pynwb.NWBFile, csd: np.ndarray,
                       times: np.ndarray, csd_virt_channel_locs: np.ndarray,
                       csd_unit="V/cm^2", position_unit="um") -> pynwb.NWBFile:
    """Add current source density (CSD) data to an nwbfile

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        nwbfile to add CSD data to
    csd : np.ndarray
        CSD data in the form of: (channels x timepoints)
    times : np.ndarray
        Timestamps for CSD data (timepoints)
    csd_virt_channel_locs : np.ndarray
        Location of interpolated channels
    csd_unit : str, optional
        Units of CSD data, by default "V/cm^2"
    position_unit : str, optional
        Units of virtual channel locations, by default "um" (micrometer)

    Returns
    -------
    pynwb.NWBFiles
        nwbfile which has had CSD data added
    """

    csd_mod = pynwb.ProcessingModule("current_source_density", "Precalculated current source density from interpolated channel locations.")
    nwbfile.add_processing_module(csd_mod)

    csd_ts = pynwb.base.TimeSeries(
        name="current_source_density",
        data=csd.T,  # TimeSeries should have data in (timepoints x channels) format
        timestamps=times,
        unit=csd_unit
    )

    x_locs, y_locs = np.split(csd_virt_channel_locs.astype(np.uint64), 2, axis=1)

    csd = EcephysCSD(name="ecephys_csd",
                     time_series=csd_ts,
                     virtual_electrode_x_positions=x_locs.flatten(),
                     virtual_electrode_x_positions__unit=position_unit,
                     virtual_electrode_y_positions=y_locs.flatten(),
                     virtual_electrode_y_positions__unit=position_unit)

    csd_mod.add_data_interface(csd)

    return nwbfile


def write_probewise_lfp_files(probes, session_id, session_metadata,
                              session_start_time, pool_size=3):

    output_paths = []

    pool = mp.Pool(processes=pool_size)
    write = partial(write_probe_lfp_file, session_id, session_metadata,
                    session_start_time, logging.getLogger("").getEffectiveLevel())

    for pout in pool.imap_unordered(write, probes):
        output_paths.append(pout)

    return output_paths


ParsedProbeData = Tuple[pd.DataFrame,  # unit_tables
                        Dict[int, np.ndarray],  # spike_times
                        Dict[int, np.ndarray],  # spike_amplitudes
                        Dict[int, np.ndarray]]  # mean_waveforms


def parse_probes_data(probes: List[Dict[str, Any]]) -> ParsedProbeData:
    """Given a list of probe dictionaries specifying data file locations, load
    and parse probe data into intermediate data structures needed for adding
    probe data to an nwbfile.

    Parameters
    ----------
    probes : List[Dict[str, Any]]
        A list of dictionaries (one entry for each probe), where each probe
        dictionary contains metadata (id, name, sampling_rate, etc...) as well
        as filepaths pointing to where probe lfp data can be found.

    Returns
    -------
    ParsedProbeData : Tuple[...]
        unit_tables : pd.DataFrame
            A table containing unit metadata from all probes.
        spike_times : Dict[int, np.ndarray]
            Keys: unit identifiers, Values: spike time arrays
        spike_amplitudes : Dict[int, np.ndarray]
            Keys: unit identifiers, Values: spike amplitude arrays
        mean_waveforms : Dict[int, np.ndarray]
            Keys: unit identifiers, Values: mean waveform arrays
    """

    unit_tables = []
    spike_times = {}
    spike_amplitudes = {}
    mean_waveforms = {}

    for probe in probes:
        unit_tables.append(pd.DataFrame(probe['units']))

        local_to_global_unit_map = {unit['cluster_id']: unit['id'] for unit in probe['units']}

        spike_times.update(read_spike_times_to_dictionary(
            probe['spike_times_path'], probe['spike_clusters_file'], local_to_global_unit_map
        ))
        mean_waveforms.update(read_waveforms_to_dictionary(
            probe['mean_waveforms_path'], local_to_global_unit_map
        ))

        spike_amplitudes.update(read_spike_amplitudes_to_dictionary(
            probe["spike_amplitudes_path"], probe["spike_clusters_file"],
            probe["templates_path"], probe["spike_templates_path"], probe["inverse_whitening_matrix_path"],
            local_to_global_unit_map=local_to_global_unit_map,
            scale_factor=probe["amplitude_scale_factor"]
        ))

    units_table = pd.concat(unit_tables).set_index(keys='id', drop=True)

    return (units_table, spike_times, spike_amplitudes, mean_waveforms)


def add_probewise_data_to_nwbfile(nwbfile, probes):
    """ Adds channel (electrode) and spike data for a single probe to the session-level nwb file.
    """
    for probe in probes:
        logging.info(f'found probe {probe["id"]} with name {probe["name"]}')

        if probe.get("temporal_subsampling_factor", None) is not None:
            probe["lfp_sampling_rate"] = probe["lfp_sampling_rate"] / probe["temporal_subsampling_factor"]

        nwbfile, probe_nwb_device, probe_nwb_electrode_group = add_probe_to_nwbfile(
            nwbfile,
            probe_id=probe["id"],
            name=probe["name"],
            sampling_rate=probe["sampling_rate"],
            lfp_sampling_rate=probe["lfp_sampling_rate"],
            has_lfp_data=probe["lfp"] is not None
        )

        add_ecephys_electrodes(nwbfile, probe["channels"], probe_nwb_electrode_group)

    units_table, spike_times, spike_amplitudes, mean_waveforms = parse_probes_data(probes)
    nwbfile.units = pynwb.misc.Units.from_dataframe(fill_df(units_table), name='units')

    sorted_spike_times, sorted_spike_amplitudes = filter_and_sort_spikes(spike_times, spike_amplitudes)

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units,
        data=sorted_spike_times,
        column_name="spike_times",
        column_description="times (s) of detected spiking events",
    )

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units,
        data=sorted_spike_amplitudes,
        column_name="spike_amplitudes",
        column_description="amplitude (s) of detected spiking events"
    )

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units,
        data=mean_waveforms,
        column_name="waveform_mean",
        column_description="mean waveforms on peak channels (and over samples)",
    )

    return nwbfile


def add_optotagging_table_to_nwbfile(nwbfile, optotagging_table, tag="optical_stimulation"):
    # "name" is a pynwb reserved column name that older versions of the
    # pre-processed optotagging_table may use.
    if "name" in optotagging_table.columns:
        optotagging_table = optotagging_table.rename(columns={"name": "stimulus_name"})

    opto_ts = pynwb.base.TimeSeries(
        name="optotagging",
        timestamps=optotagging_table["start_time"].values,
        data=optotagging_table["duration"].values,
        unit="seconds"
    )

    opto_mod = pynwb.ProcessingModule("optotagging", "optogenetic stimulution data")
    opto_mod.add_data_interface(opto_ts)
    nwbfile.add_processing_module(opto_mod)

    optotagging_table = setup_table_for_epochs(optotagging_table, opto_ts, tag)

    if len(optotagging_table) > 0:
        container = pynwb.epoch.TimeIntervals.from_dataframe(optotagging_table, "optogenetic_stimulation")
        opto_mod.add_data_interface(container)

    return nwbfile


def add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile: pynwb.NWBFile,
                                                  eye_tracking_rig_geometry: dict) -> pynwb.NWBFile:
    """ Rig geometry dict should consist of the following fields:
    monitor_position_mm: [x, y, z]
    monitor_rotation_deg: [x, y, z]
    camera_position_mm: [x, y, z]
    camera_rotation_deg: [x, y, z]
    led_position: [x, y, z]
    equipment: A string describing rig
    """
    eye_tracking_rig_mod = pynwb.ProcessingModule(name='eye_tracking_rig_metadata',
                                                  description='Eye tracking rig metadata module')

    rig_metadata = EcephysEyeTrackingRigMetadata(
        name="eye_tracking_rig_metadata",
        equipment=eye_tracking_rig_geometry['equipment'],
        monitor_position=eye_tracking_rig_geometry['monitor_position_mm'],
        monitor_position__unit="mm",
        camera_position=eye_tracking_rig_geometry['camera_position_mm'],
        camera_position__unit="mm",
        led_position=eye_tracking_rig_geometry['led_position'],
        led_position__unit="mm",
        monitor_rotation=eye_tracking_rig_geometry['monitor_rotation_deg'],
        monitor_rotation__unit="deg",
        camera_rotation=eye_tracking_rig_geometry['camera_rotation_deg'],
        camera_rotation__unit="deg"
    )

    eye_tracking_rig_mod.add_data_interface(rig_metadata)
    nwbfile.add_processing_module(eye_tracking_rig_mod)

    return nwbfile


def add_eye_tracking_data_to_nwbfile(nwbfile: pynwb.NWBFile,
                                     eye_tracking_frame_times: pd.Series,
                                     eye_dlc_tracking_data: Dict[str, pd.DataFrame],
                                     eye_gaze_data: Dict[str, pd.DataFrame]) -> pynwb.NWBFile:

    if eye_tracking_data_is_valid(eye_dlc_tracking_data=eye_dlc_tracking_data,
                                  synced_timestamps=eye_tracking_frame_times):
        add_eye_tracking_ellipse_fit_data_to_nwbfile(nwbfile,
                                                     eye_dlc_tracking_data=eye_dlc_tracking_data,
                                                     synced_timestamps=eye_tracking_frame_times)

        # --- Add gaze mapped positions to nwb file ---
        if eye_gaze_data:
            add_eye_gaze_mapping_data_to_nwbfile(nwbfile,
                                                 eye_gaze_data=eye_gaze_data)

    return nwbfile


def write_ecephys_nwb(
    output_path,
    session_id, session_start_time,
    stimulus_table_path,
    invalid_epochs,
    probes,
    running_speed_path,
    session_sync_path,
    eye_tracking_rig_geometry,
    eye_dlc_ellipses_path,
    eye_gaze_mapping_path,
    pool_size,
    optotagging_table_path=None,
    session_metadata=None,
    **kwargs
):

    nwbfile = pynwb.NWBFile(
        session_description='Data and metadata for an Ecephys session',
        identifier=f"{session_id}",
        session_id=f"{session_id}",
        session_start_time=session_start_time,
        institution="Allen Institute for Brain Science"
    )

    if session_metadata is not None:
        nwbfile = add_metadata_to_nwbfile(nwbfile, session_metadata)

    stimulus_columns_to_drop = [
        "colorSpace", "depth", "interpolate", "pos", "rgbPedestal", "tex",
        "texRes", "flipHoriz", "flipVert", "rgb", "signalDots"
    ]
    stimulus_table = read_stimulus_table(stimulus_table_path,
                                         columns_to_drop=stimulus_columns_to_drop)
    nwbfile = add_stimulus_timestamps(nwbfile, stimulus_table['start_time'].values)  # TODO: patch until full timestamps are output by stim table module
    nwbfile = add_stimulus_presentations(nwbfile, stimulus_table)
    nwbfile = add_invalid_times(nwbfile, invalid_epochs)

    if optotagging_table_path is not None:
        optotagging_table = pd.read_csv(optotagging_table_path)
        nwbfile = add_optotagging_table_to_nwbfile(nwbfile, optotagging_table)

    nwbfile = add_probewise_data_to_nwbfile(nwbfile, probes)

    running_speed, raw_running_data = read_running_speed(running_speed_path)
    add_running_speed_to_nwbfile(nwbfile, running_speed)
    add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)

    add_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile,
                                                  eye_tracking_rig_geometry)

    # Collect eye tracking/gaze mapping data from files
    eye_tracking_frame_times = su.get_synchronized_frame_times(session_sync_file=session_sync_path,
                                                               sync_line_label_keys=Dataset.EYE_TRACKING_KEYS)
    eye_dlc_tracking_data = read_eye_dlc_tracking_ellipses(Path(eye_dlc_ellipses_path))
    if eye_gaze_mapping_path:
        eye_gaze_data = read_eye_gaze_mappings(Path(eye_gaze_mapping_path))
    else:
        eye_gaze_data = None

    add_eye_tracking_data_to_nwbfile(nwbfile,
                                     eye_tracking_frame_times,
                                     eye_dlc_tracking_data,
                                     eye_gaze_data)

    Manifest.safe_make_parent_dirs(output_path)
    with pynwb.NWBHDF5IO(output_path, mode='w') as io:
        logging.info(f"writing session nwb file to {output_path}")
        io.write(nwbfile, cache_spec=True)

    probes_with_lfp = [p for p in probes if p["lfp"] is not None]
    probe_outputs = write_probewise_lfp_files(probes_with_lfp, session_id,
                                              session_metadata,
                                              session_start_time,
                                              pool_size=pool_size)

    return {
        'nwb_path': output_path,
        "probe_outputs": probe_outputs
    }


def main():
    logging.basicConfig(
        format="%(asctime)s - %(process)s - %(levelname)s - %(message)s"
    )

    parser = optional_lims_inputs(sys.argv, InputSchema, OutputSchema, get_inputs_from_lims)

    output = write_ecephys_nwb(**parser.args)
    write_or_print_outputs(output, parser)


if __name__ == "__main__":
    main()
