import logging
import sys
from pathlib import PurePath
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
)
from allensdk.brain_observatory.argschema_utilities import (
    write_or_print_outputs, optional_lims_inputs
)
from allensdk.brain_observatory import dict_to_indexed_array
from allensdk.brain_observatory.ecephys.file_io.continuous_file import ContinuousFile
from allensdk.brain_observatory.ecephys.nwb import EcephysProbe


STIM_TABLE_RENAMES_MAP = {"Start": "start_time", "End": "stop_time"}


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

    spike_times = np.squeeze(np.load(spike_times_path, allow_pickle=False))
    spike_units = np.squeeze(np.load(spike_units_path, allow_pickle=False))

    sort_order = np.argsort(spike_units)
    spike_units = spike_units[sort_order]
    spike_times = spike_times[sort_order]
    changes = np.concatenate(
        [
            np.array([0]),
            np.where(np.diff(spike_units))[0] + 1,
            np.array([spike_times.size]),
        ]
    )

    output_times = {}
    for jj, (low, high) in enumerate(zip(changes[:-1], changes[1:])):
        local_unit = spike_units[low]
        unit_times = np.sort(spike_times[low:high])

        if local_to_global_unit_map is not None:
            if local_unit not in local_to_global_unit_map:
                logging.warning(
                    f"unable to find unit at local position {local_unit} while reading spike times"
                )
                continue
            global_id = local_to_global_unit_map[local_unit]
            output_times[global_id] = unit_times
        else:
            output_times[local_unit] = unit_times

    return output_times


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


def add_probe_to_nwbfile(nwbfile, probe_id, sampling_rate, lfp_sampling_rate, description="", location=""):
    """ Creates objects required for representation of a single extracellular ephys probe within an NWB file. These objects amount 
    to a Device (this will be removed at some point from pynwb) and an ElectrodeGroup.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        file to which probe information will be assigned.
    probe_id : int
        unique identifier for this probe - will be used to fill the "name" field on this probe's device and group
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
        lfp_sampling_rate=lfp_sampling_rate
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

    nwbfile, probe_nwb_device, probe_nwb_electrode_group = add_probe_to_nwbfile(nwbfile, 
        probe_id=probe["id"], description=probe["name"], 
        sampling_rate=probe["sampling_rate"], lfp_sampling_rate=probe["lfp_sampling_rate"]
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

    lfp = pynwb.ecephys.LFP(name=f"probe_{probe['id']}_lfp")

    nwbfile.add_acquisition(lfp.create_electrical_series(
        name=f"probe_{probe['id']}_lfp_data",
        data=H5DataIO(data=lfp_data, compression='gzip', compression_opts=9),
        timestamps=H5DataIO(data=lfp_timestamps, compression='gzip', compression_opts=9),
        electrodes=electrode_table_region
    ))

    nwbfile.add_acquisition(lfp)

    csd, csd_times, csd_channels = read_csd_data_from_h5(probe["csd_path"])
    csd_channels = np.array([channel_li_id_map[li] for li in csd_channels])
    nwbfile = add_csd_to_nwbfile(nwbfile, csd, csd_times, csd_channels)

    with pynwb.NWBHDF5IO(probe['lfp']['output_path'], 'w') as lfp_writer:
        logging.info(f"writing probe lfp file to {probe['lfp']['output_path']}")
        lfp_writer.write(nwbfile)
    return {"id": probe["id"], "nwb_path": probe["lfp"]["output_path"]}


def read_csd_data_from_h5(csd_path):
    with h5py.File(csd_path, "r") as csd_file:
        return csd_file["current_source_density"][:], csd_file["timestamps"][:], csd_file["channels"][:]


def add_csd_to_nwbfile(nwbfile, csd, times, channels, unit="V/cm^2"):

    csd_mod = pynwb.ProcessingModule("current_source_density", "precalculated current source density from a subset of channel")
    nwbfile.add_processing_module(csd_mod)

    csd_ts = pynwb.base.TimeSeries(
        name="current_source_density",
        data=csd,
        timestamps=times,
        control=channels.astype(np.uint64),  # these are postgres ids, always non-negative
        control_description="ids of electrodes from which csd was calculated",
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
    mean_waveforms = {}

    for probe in probes:
        logging.info(f'found probe {probe["id"]} with name {probe["name"]}')

        nwbfile, probe_nwb_device, probe_nwb_electrode_group = add_probe_to_nwbfile(nwbfile, 
            probe_id=probe["id"], description=probe["name"], 
            sampling_rate=probe["sampling_rate"], lfp_sampling_rate=probe["lfp_sampling_rate"]
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
        data=mean_waveforms,
        column_name="waveform_mean",
        column_description="mean waveforms on peak channels (and over samples)",
    )

    return nwbfile


def write_ecephys_nwb(
    output_path, 
    session_id, session_start_time, 
    stimulus_table_path, 
    probes, 
    running_speed_path,
    pool_size,
    **kwargs
):

    nwbfile = pynwb.NWBFile(
        session_description='EcephysSession',
        identifier='{}'.format(session_id),
        session_start_time=session_start_time
    )

    stimulus_table = read_stimulus_table(stimulus_table_path)
    nwbfile = add_stimulus_timestamps(nwbfile, stimulus_table['start_time'].values) # TODO: patch until full timestamps are output by stim table module
    nwbfile = add_stimulus_presentations(nwbfile, stimulus_table)

    nwbfile = add_probewise_data_to_nwbfile(nwbfile, probes)

    running_speed, raw_running_data = read_running_speed(running_speed_path)
    add_running_speed_to_nwbfile(nwbfile, running_speed)
    add_raw_running_data_to_nwbfile(nwbfile, raw_running_data)

    Manifest.safe_make_parent_dirs(output_path)
    io = pynwb.NWBHDF5IO(output_path, mode='w')
    logging.info(f"writing session nwb file to {output_path}")
    io.write(nwbfile)
    io.close()

    probe_outputs = write_probewise_lfp_files(probes, session_start_time, pool_size=pool_size)

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
