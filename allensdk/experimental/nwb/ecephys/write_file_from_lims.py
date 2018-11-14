import argparse
from datetime import datetime
import os

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np

import pynwb
from pynwb.core import DynamicTable, VectorData, VectorIndex
from pynwb.file import ElectrodeTable
from pynwb import NWBHDF5IO
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
from pynwb.base import ProcessingModule
from pynwb.misc import Units

from allensdk.experimental.nwb.api.ecephys_lims_api import EcephysLimsApi


SESSION_START_TIME_PLACEHOLDER = datetime.now()  # TODO: this is not present in the date_of_acquisition column in lims 
CT_PLACEHOLDERS = {
    'x': -1.0,  # TODO when we get CCF positions from alignment we can write those here, till then there is no option to not supply these fields, so ...
    'y': -1.0,
    'z': -1.0,
    'imp': -1.0,  # TODO: we don't currently have this info (at least not in a form I know about)
    'location': 'null',  # TODO: again, waits on CCF registration for accurate information. Will be acronym of CCF structure
    'filtering': 'here is a description of our filtering'  # TODO do we have a standard filtering string?
}
STIM_TABLE_RENAMES_MAP = {
    'Start': 'start_time',
    'End': 'stop_time',
}


def process_channel_table_for_nwb(channel_table, placeholders=None):
    if placeholders is None:
        placeholders = CT_PLACEHOLDERS

    channel_table.set_index('id', drop=True, inplace=True) # re: ids TODO: track these in lims and use globally valid ids - till then just use local
    channel_table['group'] = None
    channel_table['group_name'] = ''

    for key, value in placeholders.items():
        channel_table[key] = value

    return channel_table


def process_stimulus_table_for_nwb(stimulus_table, renames_map=None):
    stimulus_table  = stimulus_table.rename(columns=renames_map, index={})
    stimulus_table['description'] = ''  # TODO: I guess it makes sense for epochs to have descriptions?
    stimulus_table['timeseries'] = [tuple()] * stimulus_table.shape[0]
    stimulus_table['tags'] = [tuple()] * stimulus_table.shape[0]
    
    return stimulus_table


def add_units_to_file(nwbfile, probe_table, channel_table, unit_table, spike_times):

    for _, probe in probe_table.iterrows():

        probe_nwb_device = Device(
            name=str(probe['id']), # why not name? probe names are actually codes for targeted structure. ids are the appropriate primary key
        )

        probe_nwb_electrode_group = ElectrodeGroup(
            name=str(probe['id']),
            description=probe['name'], # TODO probe name currently describes the targeting of the probe - the closest we have to a meaningful "kind"
            location='', # TODO not actailly sure where to get this
            device=probe_nwb_device
        )

        nwbfile.add_device(probe_nwb_device)
        nwbfile.add_electrode_group(probe_nwb_electrode_group)

        channel_table.loc[channel_table['probe_id'] == probe['id'], 'group'] = probe_nwb_electrode_group

    nwbfile.electrodes = ElectrodeTable().from_dataframe(channel_table, name='electrodes')

    nwbfile.add_unit_column('firing_rate', description='')
    nwbfile.add_unit_column('isi_violations', description='')
    nwbfile.add_unit_column('peak_channel_id', description='')
    nwbfile.add_unit_column('probe_id', description='')
    nwbfile.add_unit_column('quality', description='')
    nwbfile.add_unit_column('snr', description='')

    for unit_id, unit_row in unit_table.iterrows():
        nwbfile.add_unit(
            id=unit_id,
            spike_times=spike_times[unit_id],
            **unit_row
        )
    return nwbfile


def build_file(ecephys_session_id):
    api = EcephysLimsApi()

    session_info = api.get_session_table(session_ids=[ecephys_session_id]).to_dict('record')[0]
    # stimulus_table = process_stimulus_table_for_nwb(api.get_stimulus_table(ecephys_session_id))
    probe_table = api.get_probe_table(session_ids=[ecephys_session_id])
    channel_table = process_channel_table_for_nwb(api.get_channel_table(ecephys_session_id))  # TODO: 706875901 breaks here
    unit_table = api.get_unit_table(session_id=ecephys_session_id)
    unit_table.set_index('id', drop=True, inplace=True)
    spike_times = api.get_spike_times(ecephys_session_id)

    nwbfile = pynwb.NWBFile(
        session_description='EcephysSession',
        identifier='{}'.format(ecephys_session_id),
        session_start_time=SESSION_START_TIME_PLACEHOLDER,
        file_create_date=datetime.now()
    )

    nwbfile = add_units_to_file(nwbfile, probe_table, channel_table, unit_table, spike_times)
    # nwbfile.epochs = Epochs.from_dataframe(stimulus_table)  # TODO: I can't actually find an experiment that both as a stim table and has the current format for its other data
    # nwbfile = add_spike_times_to_file(nwbfile, )
    return nwbfile


def main(ecephys_session_id, nwb_path, remove_file=False):

    nwbfile = build_file(ecephys_session_id)

    if remove_file:
        os.remove(nwb_path)

    io = NWBHDF5IO(nwb_path, mode='w')
    io.write(nwbfile)
    io.close()

    return nwbfile, nwb_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ecephys_session_id', type=int) # 754312389, for instance ... TODO: 754312389 ONLY
    parser.add_argument('--nwb_path', type=str, default=None)
    parser.add_argument('--remove_file', action='store_true', default=False)

    args = parser.parse_args()
    if args.nwb_path is None:
        nwb_path = '{}.nwb'.format(args.ecephys_session_id)
    else:
        nwb_path = args.nwb_path

    main(args.ecephys_session_id, nwb_path, args.remove_file)