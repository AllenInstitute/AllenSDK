import logging
import sys
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
from allensdk.brain_observatory.nwb import (
    add_stimulus_presentations,
    add_stimulus_timestamps,
    add_invalid_times,
    setup_table_for_epochs,
    setup_table_for_invalid_times,
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
from allensdk.brain_observatory.ecephys.nwb import EcephysProbe, EcephysLabMetaData
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


def read_stimulus_table(path, column_renames_map=None):
    """ Loads from a CSV on disk the stimulus table for this session. Optionally renames columns to match NWB 
    epoch specifications.

    Parameters
    ----------
    path : str
        path to stimulus table csv
    column_renames_map : dict, optional
        if provided will be used to rename columns from keys -> values. Default renames 'Start' -> 'start_time' and 
        'End' -> 'stop_time'
    
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
        templates[temp_idx,:,:] = np.dot(
            np.ascontiguousarray(templates[temp_idx,:,:]), 
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


def add_metadata_to_nwbfile(nwbfile, metadata):
    nwbfile.add_lab_meta_data(
        EcephysLabMetaData(name="metadata", **metadata)
    )
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


def add_probe_to_nwbfile(nwbfile, probe_id, sampling_rate, lfp_sampling_rate, has_lfp_data,
                         description="",
                         location=""):
    """ Creates objects required for representation of a single extracellular ephys probe within an NWB file. These objects amount 
    to a Device (this will be removed at some point from pynwb) and an ElectrodeGroup.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        file to which probe information will be assigned.
    probe_id : int
        unique identifier for this probe - will be used to fill the "name" field on this probe's device and group
    sampling_rate: float,
        sampling rate
    lfp_sampling_rate: float
        sampling rate of LFP
    has_lfp_data: bool
        True if LFP data is available for the probe, otherwise False
    description : str, optional
        human-readable description of this probe. Practically (and temporarily), we use tags like "probeA" or "probeB"
    location : str, optional
        unspecified information about the location of this probe. Currently left blank, but in the future will probably contain
        an expanded form of the information currently implicit in 'probeA' (ie targeting and insertion plan) while channels will 
        carry the results of ccf registration.

    Returns
    ------
        nwbfile : pynwb.NWBFile
            the updated file object
        probe_nwb_device : pynwb.device.Device
            device object corresponding to this probe
        probe_nwb_electrode_group : pynwb.ecephys.ElectrodeGroup
            electrode group object corresponding to this probe

    """
    probe_nwb_device = pynwb.device.Device(name=str(probe_id))
    probe_nwb_electrode_group = EcephysProbe(
        name=str(probe_id),
        description=description,
        location=location,
        device=probe_nwb_device,
        sampling_rate=sampling_rate,
        lfp_sampling_rate=lfp_sampling_rate,
        has_lfp_data=has_lfp_data,
    )

    nwbfile.add_device(probe_nwb_device)
    nwbfile.add_electrode_group(probe_nwb_electrode_group)

    return nwbfile, probe_nwb_device, probe_nwb_electrode_group


def prepare_probewise_channel_table(channels, electrode_group):
    """ Builds an NWB-ready dataframe of probewise channels

    Parameters
    ----------
    channels : pd.DataFrame
        each row is a channel. ids may be listed as a column or index named "id". Other expected columns:
            probe_id: int, uniquely identifies probe
            valid_data: bool, whether data from this channel is usable
            local_index: uint, the probe-local index of this channel
            probe_vertical_position: the lengthwise position along the probe of this channel (microns)
            probe_horizontal_position: the widthwise position on the probe of this channel (microns)
    electrode_group : pynwb.ecephys.ElectrodeGroup
        the electrode group representing this probe's electrodes

    Returns
    -------
    channel_table : pd.DataFrame
        probewise channel table, ready to be concatenated and passed to ElectrodeTable.from_dataframe

    """

    channel_table = pd.DataFrame(channels).copy()

    if "id" in channel_table.columns and channel_table.index.name != "id":
        channel_table = channel_table.set_index(keys="id", drop=True)
    elif "id" in channel_table.columns and channel_table.index.name == "id":
        raise ValueError('found both column and index named "id"')
    elif channel_table.index.name != "id":
        raise ValueError(
            f"unable to recognize ids in this channel table. index: {channel_table.index}, columns: {channel_table.columns}"
        )

    channel_table["group"] = electrode_group
    return channel_table


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
        timestamps=np.array([
            running_speed["start_time"].values, 
            running_speed["end_time"].values
        ]),
        data=running_speed["velocity"].values,
        unit=units["velocity"]
    )

    rotation_timeseries = pynwb.base.TimeSeries(
        name="running_wheel_rotation",
        timestamps=running_speed_timeseries,
        data=running_speed["net_rotation"].values,
        unit=units["rotation"]
    )

    running_mod.add_data_interface(running_speed_timeseries)
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


def write_probe_lfp_file(session_start_time, log_level, probe):
    """ Writes LFP data (and associated channel information) for one probe to a standalone nwb file
    """

    logging.getLogger('').setLevel(log_level)
    logging.info(f"writing lfp file for probe {probe['id']}")

    nwbfile = pynwb.NWBFile(
        session_description='EcephysProbe',
        identifier=f"{probe['id']}",
        session_start_time=session_start_time
    )    


    if probe.get("temporal_subsampling_factor", None) is not None:
        probe["lfp_sampling_rate"] = probe["lfp_sampling_rate"] / probe["temporal_subsampling_factor"]

    nwbfile, probe_nwb_device, probe_nwb_electrode_group = add_probe_to_nwbfile(
        nwbfile,
        probe_id=probe["id"],
        description=probe["name"],
        sampling_rate=probe["sampling_rate"],
        lfp_sampling_rate=probe["lfp_sampling_rate"],
        has_lfp_data=probe["lfp"] is not None
    )

    channels = prepare_probewise_channel_table(probe['channels'], probe_nwb_electrode_group)
    channel_li_id_map = {row["local_index"]: cid for cid, row in channels.iterrows()}
    lfp_channels = np.load(probe['lfp']['input_channels_path'], allow_pickle=False)
    
    channels.reset_index(inplace=True)
    channels.set_index("local_index", inplace=True)
    channels = channels.loc[lfp_channels, :]
    channels.reset_index(inplace=True)
    channels.set_index("id", inplace=True)

    channels = fill_df(channels)
    
    nwbfile.electrodes = pynwb.file.ElectrodeTable().from_dataframe(channels, name='electrodes')
    electrode_table_region = nwbfile.create_electrode_table_region(
        region=np.arange(channels.shape[0]).tolist(),  # must use raw indices here
        name='electrodes',
        description=f"lfp channels on probe {probe['id']}"
    )

    lfp_data, lfp_timestamps = ContinuousFile(
        data_path=probe['lfp']['input_data_path'],
        timestamps_path=probe['lfp']['input_timestamps_path'],
        total_num_channels=channels.shape[0]
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
        lfp_writer.write(nwbfile)
    return {"id": probe["id"], "nwb_path": probe["lfp"]["output_path"]}


def read_csd_data_from_h5(csd_path):
    with h5py.File(csd_path, "r") as csd_file:
        return (csd_file["current_source_density"][:],
                csd_file["timestamps"][:],
                csd_file["csd_locations"][:])


def add_csd_to_nwbfile(nwbfile, csd, times, csd_virt_channel_locs, unit="V/cm^2"):

    csd_mod = pynwb.ProcessingModule("current_source_density", "Precalculated current source density from interpolated channel locations.")
    nwbfile.add_processing_module(csd_mod)

    csd_ts = pynwb.base.TimeSeries(
        name="current_source_density",
        data=csd,
        timestamps=times,
        control=csd_virt_channel_locs.astype(np.uint64),  # These are locations (x, y) of virtual interpolated electrodes
        control_description="Virtual locations of electrodes from which csd was calculated",
        unit=unit
    )
    csd_mod.add_data_interface(csd_ts)

    return nwbfile


def write_probewise_lfp_files(probes, session_start_time, pool_size=3):

    output_paths = []

    pool = mp.Pool(processes=pool_size)
    write = partial(write_probe_lfp_file, session_start_time, logging.getLogger("").getEffectiveLevel())

    for pout in pool.imap_unordered(write, probes):
        output_paths.append(pout)

    return output_paths
    

def add_probewise_data_to_nwbfile(nwbfile, probes):
    """ Adds channel and spike data for a single probe to the session-level nwb file.
    """

    channel_tables = {}
    unit_tables = []
    spike_times = {}
    spike_amplitudes = {}
    mean_waveforms = {}

    for probe in probes:
        logging.info(f'found probe {probe["id"]} with name {probe["name"]}')

        if probe.get("temporal_subsampling_factor", None) is not None:
            probe["lfp_sampling_rate"] = probe["lfp_sampling_rate"] / probe["temporal_subsampling_factor"]

        nwbfile, probe_nwb_device, probe_nwb_electrode_group = add_probe_to_nwbfile(
            nwbfile,
            probe_id=probe["id"],
            description=probe["name"],
            sampling_rate=probe["sampling_rate"],
            lfp_sampling_rate=probe["lfp_sampling_rate"],
            has_lfp_data=probe["lfp"] is not None
        )

        channel_tables[probe["id"]] = prepare_probewise_channel_table(probe['channels'], probe_nwb_electrode_group)
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
    
    electrodes_table = fill_df(pd.concat(list(channel_tables.values())))
    nwbfile.electrodes = pynwb.file.ElectrodeTable().from_dataframe(electrodes_table, name='electrodes')
    units_table = pd.concat(unit_tables).set_index(keys='id', drop=True)
    nwbfile.units = pynwb.misc.Units.from_dataframe(fill_df(units_table), name='units')

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units,
        data=spike_times,
        column_name="spike_times",
        column_description="times (s) of detected spiking events",
    )

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units,
        data=spike_amplitudes,
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
    opto_ts = pynwb.base.TimeSeries(
        name="optotagging",
        timestamps=optotagging_table["start_time"].values,
        data=optotagging_table["duration"].values
    )

    opto_mod = pynwb.ProcessingModule("optotagging", "optogenetic stimulution data")
    opto_mod.add_data_interface(opto_ts)
    nwbfile.add_processing_module(opto_mod)

    optotagging_table = setup_table_for_epochs(optotagging_table, opto_ts, tag)
    
    if len(optotagging_table) > 0:
        container = pynwb.epoch.TimeIntervals.from_dataframe(optotagging_table, "optogenetic_stimuluation")
        opto_mod.add_data_interface(container)

    return nwbfile


def append_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile: pynwb.NWBFile,
                                                     eye_tracking_rig_geometry: dict) -> pynwb.NWBFile:
    """ Rig geometry dict should consist of the following fields:
    monitor_position_mm: [x, y, z]
    monitor_rotation_deg: [x, y, z]
    camera_position_mm: [x, y, z]
    camera_rotation_deg: [x, y, z]
    led_position_mm: [x, y, z]
    equipment: A string describing rig
    """
    rig_geometry_data = pd.DataFrame(eye_tracking_rig_geometry,
                                     index=['x', 'y', 'z']).drop('equipment', axis=1)
    rig_geometry_data['pynwb_index'] = range(len(rig_geometry_data))
    equipment_data = pd.DataFrame({"equipment": eye_tracking_rig_geometry['equipment']}, index=[0])

    eye_tracking_mod = nwbfile.modules['eye_tracking']
    rig_geometry_interface = pynwb.core.DynamicTable.from_dataframe(df=rig_geometry_data,
                                                                    name="rig_geometry_data",
                                                                    index_column='pynwb_index')
    equipment_interface = pynwb.core.DynamicTable.from_dataframe(df=equipment_data,
                                                                 name="equipment")
    eye_tracking_mod.add_data_interface(rig_geometry_interface)
    eye_tracking_mod.add_data_interface(equipment_interface)

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
        session_description='EcephysSession',
        identifier='{}'.format(session_id),
        session_start_time=session_start_time
    )

    if session_metadata is not None:
        nwbfile = add_metadata_to_nwbfile(nwbfile, session_metadata)

    stimulus_table = read_stimulus_table(stimulus_table_path)
    nwbfile = add_stimulus_timestamps(nwbfile, stimulus_table['start_time'].values) # TODO: patch until full timestamps are output by stim table module
    nwbfile = add_stimulus_presentations(nwbfile, stimulus_table)
    nwbfile = add_invalid_times(nwbfile,invalid_epochs)

    if optotagging_table_path is not None:
        optotagging_table = pd.read_csv(optotagging_table_path)
        nwbfile = add_optotagging_table_to_nwbfile(nwbfile, optotagging_table)

    nwbfile = add_probewise_data_to_nwbfile(nwbfile, probes)

    running_speed, raw_running_data = read_running_speed(running_speed_path)
    add_running_speed_to_nwbfile(nwbfile, running_speed)
    add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)

    # --- Add eye tracking ellipse fits to nwb file ---
    eye_tracking_frame_times = su.get_synchronized_frame_times(session_sync_file=session_sync_path,
                                                               sync_line_label_keys=Dataset.EYE_TRACKING_KEYS)
    eye_dlc_tracking_data = read_eye_dlc_tracking_ellipses(Path(eye_dlc_ellipses_path))

    if eye_tracking_data_is_valid(eye_dlc_tracking_data=eye_dlc_tracking_data,
                                  synced_timestamps=eye_tracking_frame_times):
        add_eye_tracking_ellipse_fit_data_to_nwbfile(nwbfile,
                                                     eye_dlc_tracking_data=eye_dlc_tracking_data,
                                                     synced_timestamps=eye_tracking_frame_times)

        # --- Append eye tracking rig geometry info to nwb file (with eye tracking) ---
        append_eye_tracking_rig_geometry_data_to_nwbfile(nwbfile,
                                                         eye_tracking_rig_geometry=eye_tracking_rig_geometry)

        # --- Add gaze mapped positions to nwb file ---
        if eye_gaze_mapping_path:
            eye_gaze_data = read_eye_gaze_mappings(Path(eye_gaze_mapping_path))
            add_eye_gaze_mapping_data_to_nwbfile(nwbfile,
                                                 eye_gaze_data=eye_gaze_data)

    Manifest.safe_make_parent_dirs(output_path)
    io = pynwb.NWBHDF5IO(output_path, mode='w')
    logging.info(f"writing session nwb file to {output_path}")
    io.write(nwbfile)
    io.close()

    probes_with_lfp = [p for p in probes if p["lfp"] is not None]
    probe_outputs = write_probewise_lfp_files(probes_with_lfp, session_start_time, pool_size=pool_size)

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
