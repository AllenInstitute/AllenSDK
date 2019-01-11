import os

import numpy as np
import pandas as pd
import pynwb


STIM_TABLE_RENAMES_MAP = {
    'Start': 'start_time',
    'End': 'stop_time',
}


def add_probes_to_file(nwbfile, probes):

    probe_object_map = {}

    for probe in probes:
        probe_nwb_device = pynwb.Device(
            name=str(probe['id']), # why not name? probe names are actually codes for targeted structure. ids are the appropriate primary key
        )
        probe_nwb_electrode_group = pynwb.ElectrodeGroup(
            name=str(probe['id']),
            description=probe['name'], # TODO probe name currently describes the targeting of the probe - the closest we have to a meaningful "kind"
            location='', # TODO not actailly sure where to get this
            device=probe_nwb_device
        )

        nwbfile.add_device(probe_nwb_device)
        nwbfile.add_electrode_group(probe_nwb_electrode_group)
        probe_object_map[probe['id']] = probe_nwb_electrode_group

    return nwbfile, probe_object_map


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