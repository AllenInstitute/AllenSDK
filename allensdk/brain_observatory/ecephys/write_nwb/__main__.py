import logging
import sys
import argparse
from datetime import datetime
import os

import marshmallow
import argschema
import pynwb
import requests
import pandas as pd
import numpy as np

from allensdk.config.manifest import Manifest
from allensdk.brain_observatory.running_speed import RunningSpeed

from allensdk.brain_observatory.nwb._schemas import InputSchema, OutputSchema
from allensdk.brain_observatory.argschema_utilities import write_or_print_outputs


STIM_TABLE_RENAMES_MAP = {
    'Start': 'start_time',
    'End': 'stop_time',
}


def get_inputs_from_lims(host, ecephys_session_id, output_root, job_queue, strategy):
    ''' This is a development / testing utility for running this module from the Allen Institute for Brain Science's 
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

    '''
    
    uri = f'{host}/input_jsons?object_id={ecephys_session_id}&object_class=EcephysSession&strategy_class={strategy}&job_queue_name={job_queue}&output_directory={output_root}'
    response = requests.get(uri)
    data = response.json()

    if len(data) == 1 and 'error' in data:
        raise ValueError('bad request uri: {} ({})'.format(uri, data['error']))

    return data 


def read_stimulus_table(path,  column_renames_map=None):
    ''' Loads from a CSV on disk the stimulus table for this session. Optionally renames columns to match NWB 
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

    '''

    if column_renames_map  is None:
        column_renames_map = STIM_TABLE_RENAMES_MAP

    _, ext = os.path.splitext(path)

    if ext == '.csv':
        stimulus_table = pd.read_csv(path)
    else:
        raise IOError(f'unrecognized stimulus table extension: {ext}')

    return stimulus_table.rename(columns=column_renames_map, index={})


def add_stimulus_table_to_file(nwbfile, stimulus_table, tag='stimulus_epoch'):
    ''' Adds a stimulus table (defining stimulus characteristics for each time point in a session) to an nwbfile as epochs.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
    stimulus_table: pd.DataFrame
        Each row corresponds to an epoch of time. Columns define the epoch (start and stop time) and its characteristics. 
        Nans will be replaced with the empty string. Required columns are:
            start_time :: the time at which this epoch started
            stop_time :: the time  at which this epoch ended
    tag : str, optional
        Each epoch in an nwb file has one or more tags. This string will be applied as a tag to all epochs created here

    Returns
    -------
    nwbfile : pynwb.NWBFile

    '''
    stimulus_table = stimulus_table.copy()

    ts = pynwb.base.TimeSeries(
        name='stimulus_times', 
        timestamps=stimulus_table['start_time'].values, 
        data=stimulus_table['stop_time'].values - stimulus_table['start_time'].values,
        unit='s',
        description='start times (timestamps) and durations (data) of stimulus presentation epochs'
    )
    nwbfile.add_acquisition(ts)

    for colname, series in stimulus_table.items():
        types = set(series.map(type))
        if len(types) > 1 and str in types:
            series.fillna('', inplace=True)
            stimulus_table[colname] = series.transform(str)

    stimulus_table['tags'] = [(tag,)] * stimulus_table.shape[0]
    stimulus_table['timeseries'] = [(ts,)] * stimulus_table.shape[0]

    container = pynwb.epoch.TimeIntervals.from_dataframe(stimulus_table, 'epochs')
    nwbfile.epochs = container

    return nwbfile


def read_spike_times_to_dictionary(spike_times_path, spike_units_path, local_to_global_unit_map=None):
    ''' Reads spike times and assigned units from npy files into a lookup table.

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

    '''

    spike_times = np.squeeze(np.load(spike_times_path, allow_pickle=False))
    spike_units = np.squeeze(np.load(spike_units_path, allow_pickle=False))

    sort_order = np.argsort(spike_units)
    spike_units = spike_units[sort_order]
    spike_times = spike_times[sort_order]
    changes = np.concatenate([np.array([0]), np.where(np.diff(spike_units))[0] + 1, np.array([spike_times.size])])

    output_times = {}
    for jj, (low, high) in enumerate(zip(changes[:-1], changes[1:])):
        local_unit = spike_units[low]
        unit_times = np.sort(spike_times[low:high])

        if local_to_global_unit_map is not None:
            if local_unit not in local_to_global_unit_map:
                logging.warning(f'unable to find unit at local position {local_unit} while reading spike times')
                continue
            global_id = local_to_global_unit_map[local_unit]
            output_times[global_id] = unit_times
        else:
            output_times[local_unit] = unit_times

    return output_times


def read_waveforms_to_dictionary(waveforms_path, local_to_global_unit_map=None, peak_channel_map=None):
    ''' Builds a lookup table for unitwise waveform data

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

    '''

    waveforms = np.squeeze(np.load(waveforms_path, allow_pickle=False))
    output_waveforms = {}
    for unit_id, waveform in enumerate(np.split(waveforms, waveforms.shape[0], axis=0)):
        if local_to_global_unit_map is not None:
            if unit_id not in local_to_global_unit_map:
                logging.warning(f'unable to find unit at local position {unit_id} while reading waveforms')
                continue
            unit_id = local_to_global_unit_map[unit_id]

        if peak_channel_map is not None:
            waveform = waveform[:, peak_channel_map[unit_id]]
        
        output_waveforms[unit_id] = np.squeeze(waveform)

    return output_waveforms


def read_running_speed(values_path, timestamps_path):
    ''' Reads running speed data and timestamps into a RunningSpeed named tuple

    Parameters
    ----------
    values_path : str
        path to npy file containing running speed values (cm / s) per sample
    timestamps_path : str
        path to npy file identifying the time at which each running speed sample was collected


    Returns
    -------
    RunningSpeed : 
        contains values and timestamps

    '''

    return RunningSpeed(
        timestamps=np.squeeze(np.load(timestamps_path, allow_pickle=False)),
        values=np.squeeze(np.load(values_path, allow_pickle=False))
    )


def add_probe_to_nwbfile(nwbfile, probe_id, description='', location=''):
    ''' Creates objects required for representation of a single extracellular ephys probe within an NWB file. These objects amount 
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

    '''

    probe_nwb_device = pynwb.device.Device(name=str(probe_id))
    probe_nwb_electrode_group = pynwb.ecephys.ElectrodeGroup(
        name=str(probe_id),
        description=description,
        location=location,
        device=probe_nwb_device
    )
    
    nwbfile.add_device(probe_nwb_device)
    nwbfile.add_electrode_group(probe_nwb_electrode_group)

    return nwbfile, probe_nwb_device, probe_nwb_electrode_group


def prepare_probewise_channel_table(channels, electrode_group):
    ''' Builds an NWB-ready dataframe of probewise channels

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

    '''

    channel_table = pd.DataFrame(channels).copy()

    if 'id' in channel_table.columns and channel_table.index.name != 'id':
        channel_table = channel_table.set_index(keys='id', drop=True)
    elif 'id' in channel_table.columns and channel_table.index.name == 'id':
        raise ValueError('found both column and index named \"id\"')
    elif channel_table.index.name != 'id':
        raise ValueError(f'unable to recognize ids in this channel table. index: {channel_table.index}, columns: {channel_table.columns}')

    channel_table['group'] = electrode_group
    return channel_table


def dict_to_indexed_array(dc, order=None):
    ''' Given a dictionary and an ordered arr, build a concatenation of the dictionary's values and an index describing 
    how that concatenation can be unpacked
    '''

    if order is None:
        order = dc.keys()

    data = []
    index = []
    counter = 0

    for ii, key in enumerate(order):
        data.append(dc[key])
        counter += len(dc[key])
        index.append(counter)

    data = np.concatenate(data)

    return index, data


def add_ragged_data_to_dynamic_table(table, data, column_name, column_description=''):
    ''' Builds the index and data vectors required for writing ragged array data to a pynwb dynamic table

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

    '''

    idx, values = dict_to_indexed_array(data, table.id)
    del data

    table.add_column(name=column_name, description=column_description, data=values, index=idx)


def add_running_speed_to_nwbfile(nwbfile, running_speed, name='running_speed', unit='cm/s'):
    ''' Adds running speed data to an NWBFile as a timeseries in acquisition

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        File to which runnign speeds will be written
    running_speed : RunningSpeed
        Contains attributes 'values' and 'timestamps'
    name : str, optional
        used as name of timeseries object
    unit : str, optional
        SI units of running speed values

    Returns
    -------
    nwbfile : pynwb.NWBFile

    '''

    running_speed_series = pynwb.base.TimeSeries(
        name=name, 
        data=running_speed.values, 
        timestamps=running_speed.timestamps, 
        unit=unit
    )
    nwbfile.add_acquisition(running_speed_series)
    return nwbfile


def write_ecephys_nwb(
    output_path, 
    session_id, session_start_time, 
    stimulus_table_path, 
    probes, 
    running_speed, 
    **kwargs
):

    nwbfile = pynwb.NWBFile(
        session_description='EcephysSession',
        identifier='{}'.format(session_id),
        session_start_time=session_start_time
    )

    stimulus_table = read_stimulus_table(stimulus_table_path)
    nwbfile = add_stimulus_table_to_file(nwbfile, stimulus_table)

    channel_tables = []
    unit_tables = []
    spike_times = {}
    mean_waveforms = {}

    for probe in probes:
        logging.info(f'found probe {probe["id"]} with name {probe["name"]}')

        nwbfile, probe_nwb_device, probe_nwb_electrode_group = add_probe_to_nwbfile(nwbfile, probe['id'], location=probe['name'])
        channel_tables.append(prepare_probewise_channel_table(probe['channels'], probe_nwb_electrode_group))
        unit_tables.append(pd.DataFrame(probe['units']))

        local_to_global_unit_map = {unit['local_index']: unit['id'] for unit in probe['units']}

        spike_times.update(read_spike_times_to_dictionary(
            probe['spike_times_path'], probe['spike_clusters_file'], local_to_global_unit_map
        ))
        mean_waveforms.update(read_waveforms_to_dictionary(
            probe['mean_waveforms_path'], local_to_global_unit_map
        ))

    nwbfile.electrodes = pynwb.file.ElectrodeTable().from_dataframe(pd.concat(channel_tables), name='electrodes')
    units_table = pd.concat(unit_tables).set_index(keys='id', drop=True)
    nwbfile.units = pynwb.misc.Units.from_dataframe(units_table, name='units')

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units, 
        data=spike_times, 
        column_name='spike_times', 
        column_description='times (s) of detected spiking events'
    )

    add_ragged_data_to_dynamic_table(
        table=nwbfile.units, 
        data=mean_waveforms, 
        column_name='waveform_mean', 
        column_description='mean waveforms on peak channels (and over samples)'
    )

    running_speed = read_running_speed(running_speed['running_speed_path'], running_speed['running_speed_timestamps_path'])
    add_running_speed_to_nwbfile(nwbfile, running_speed)
    del running_speed

    Manifest.safe_make_parent_dirs(output_path)
    io = pynwb.NWBHDF5IO(output_path, mode='w')
    io.write(nwbfile)
    io.close()

    return {'nwb_path': output_path}


def main():
    logging.basicConfig(format='%(asctime)s - %(process)s - %(levelname)s - %(message)s')

    remaining_args = sys.argv[1:]
    input_data = {}
    if '--get_inputs_from_lims' in sys.argv:
        lims_parser = argparse.ArgumentParser(add_help=False)
        lims_parser.add_argument('--host', type=str, default='http://lims2')
        lims_parser.add_argument('--job_queue', type=str, default=None)
        lims_parser.add_argument('--strategy', type=str,default= None)
        lims_parser.add_argument('--ecephys_session_id', type=int, default=None)
        lims_parser.add_argument('--output_root', type=str, default= None)

        lims_args, remaining_args = lims_parser.parse_known_args(remaining_args)
        remaining_args = [item for item in remaining_args if item != '--get_inputs_from_lims']
        input_data = get_inputs_from_lims(**lims_args.__dict__)


    try:
        parser = argschema.ArgSchemaParser(
            args=remaining_args,
            input_data=input_data,
            schema_type=InputSchema,
            output_schema_type=OutputSchema,
        )
    except marshmallow.exceptions.ValidationError as err:
        print(input_data)
        raise

    output = write_ecephys_nwb(**parser.args)
    write_or_print_outputs(output, parser)


if __name__ == '__main__':
    main()
