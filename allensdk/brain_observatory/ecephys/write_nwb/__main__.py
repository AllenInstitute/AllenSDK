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

from ._schemas import InputSchema, OutputSchema


def get_inputs_from_lims(host, ecephys_session_id, output_root, job_queue, strategy):
    
    uri = f'{host}/input_jsons?object_id={ecephys_session_id}&object_class=EcephysSession&strategy_class={strategy}&job_queue_name={job_queue}&output_directory={output_root}'
    response = requests.get(uri)
    data = response.json()

    if len(data) == 1 and 'error' in data:
        raise ValueError('bad request uri: {} ({})'.format(uri, data['error']))

    return data


def write_or_print_outputs(data, parser):
    data.update({'input_parameters': parser.args})
    if 'output_json' in parser.args:
        parser.output(data, indent=2)
    else:
        print(parser.get_output_json(data))   


STIM_TABLE_RENAMES_MAP = {
    'Start': 'start_time',
    'End': 'stop_time',
}


def add_probes_to_file(nwbfile, probes):

    probe_object_map = {}

    for probe in probes:
        probe_nwb_device = pynwb.device.Device(
            name=str(probe['id']), # why not name? probe names are actually codes for targeted structure. ids are the appropriate primary key
        )
        probe_nwb_electrode_group = pynwb.ecephys.ElectrodeGroup(
            name=str(probe['id']),
            description=probe['name'], # TODO probe name currently describes the targeting of the probe - the closest we have to a meaningful "kind"
            location='', # TODO not actailly sure where to get this
            device=probe_nwb_device
        )

        nwbfile.add_device(probe_nwb_device)
        nwbfile.add_electrode_group(probe_nwb_electrode_group)
        probe_object_map[probe['id']] = probe_nwb_electrode_group

    return nwbfile, probe_object_map


def add_electrodes_to_file(nwbfile, channel_table, probe_object_map):

    for probe_id, probe_group in probe_object_map.items():
        channel_table.loc[channel_table['probe_id'] == probe['id'], 'group'] = probe_group

    nwbfile.electrodes = pynwb.ElectrodeTable().from_dataframe(channel_table, name='electrodes')
    return nwbfile


def read_stimulus_table(path,  column_renames_map=None):
    if column_renames_map  is None:
        column_renames_map = STIM_TABLE_RENAMES_MAP

    _, ext = os.path.splitext(path)

    if ext == '.csv':
        stimulus_table = pd.read_csv(path)
    else:
        raise IOError(f'unrecognized stimulus table extension: {ext}')

    return stimulus_table.rename(columns=column_renames_map, index={})


def add_stimulus_table_to_file(nwbfile, stimulus_table):

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

    stimulus_table['tags'] = [('stimulus_epoch',)] * stimulus_table.shape[0]
    stimulus_table['timeseries'] = [(ts,)] * stimulus_table.shape[0]

    container = pynwb.epoch.TimeIntervals.from_dataframe(stimulus_table, 'epochs')
    nwbfile.epochs = container

    return nwbfile


def read_spike_times_to_dictionary(spike_times_path, spike_units_path, local_to_global_unit_map=None):

    spike_times = np.squeeze(np.load(spike_times_path, allow_pickle=False))
    spike_units = np.squeeze(np.load(spike_units_path, allow_pickle=False))

    sort_order = np.argsort(spike_units)
    spike_units = spike_units[sort_order]
    spike_times = spike_times[sort_order]
    changes = np.concatenate([np.array([0]), np.where(np.diff(spike_units))[0] + 1, np.array([spike_times.size])])

    output_times = {}
    for jj, (low, high) in enumerate(zip(changes[:-1], changes[1:])):
        local_unit = spike_units[low]
        unit_times = spike_times[low:high]

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


def dict_to_indexed_array(dc, order=None):

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

        probe_nwb_device = pynwb.device.Device(
            name=str(probe['id']), # why not name? probe names are actually codes for targeted structure. ids are the appropriate primary key
        )
        probe_nwb_electrode_group = pynwb.ecephys.ElectrodeGroup(
            name=str(probe['id']),
            description=probe['name'], # TODO probe name currently describes the targeting of the probe - the closest we have to a meaningful "kind"
            location='', # TODO not actailly sure where to get this
            device=probe_nwb_device
        )
        
        nwbfile.add_device(probe_nwb_device)
        nwbfile.add_electrode_group(probe_nwb_electrode_group)

        global_to_local_channel_map = {ch['id']: ch['local_index'] for ch in probe['channels']}
        probewise_channel_table = pd.DataFrame(probe['channels'])
        probewise_channel_table['group'] = probe_nwb_electrode_group
        channel_tables.append(probewise_channel_table)

        local_to_global_unit_id_map = {unit['local_index']: unit['id'] for unit in probe['units']}

        probewise_units_table = pd.DataFrame(probe['units'])
        probewise_spike_times = read_spike_times_to_dictionary(
            probe['spike_times_path'], probe['spike_clusters_file'], local_to_global_unit_id_map
        )
        probewise_mean_waveforms = read_waveforms_to_dictionary(
            probe['mean_waveforms_path'], local_to_global_unit_id_map
        )

        unit_tables.append(probewise_units_table)
        spike_times.update(probewise_spike_times)
        mean_waveforms.update(probewise_mean_waveforms)

    channel_table = pd.concat(channel_tables)
    channel_table = channel_table.set_index(keys='id', drop=True)

    unit_table = pd.concat(unit_tables)
    unit_table = unit_table.set_index(keys='id', drop=True)

    nwbfile.electrodes = pynwb.file.ElectrodeTable().from_dataframe(channel_table, name='electrodes')
    nwbfile.units = pynwb.misc.Units.from_dataframe(unit_table, name='units')

    unit_id_order = unit_table.index.values.copy()
    sindex, stimes = dict_to_indexed_array(spike_times, unit_id_order)
    del spike_times
    nwbfile.units.add_column(name='spike_times', description='times (s) of detected spiking events', data=stimes, index=sindex)
    del stimes
    del sindex

    mwindex, mwdata = dict_to_indexed_array(mean_waveforms, unit_id_order)
    del mean_waveforms
    nwbfile.units.add_column(name='waveform_mean', description='mean waveforms on peak channels (and over samples) for each unit', data=mwdata, index=mwindex)
    del mwdata
    del mwindex

    running_speeds = np.squeeze(np.load(running_speed['running_speed_path'], allow_pickle=False))
    running_speed_timestamps = np.squeeze(np.load(running_speed['running_speed_timestamps_path'], allow_pickle=False))
    running_speed_series = pynwb.base.TimeSeries(name='running_speed', data=running_speeds, timestamps=running_speed_timestamps, unit='cm/s')
    del running_speeds
    del running_speed_timestamps
    nwbfile.add_acquisition(running_speed_series)

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
