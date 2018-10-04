import argparse
from datetime import datetime
import os

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np

import pynwb
from pynwb.core import DynamicTable
from pynwb.file import ElectrodeTable
from pynwb import NWBHDF5IO
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup

from allensdk.experimental.nwb.ecephys.api.ecephys_lims_api import EcephysLimsApi


DEFAULT_DATABASE = 'lims2'
DEFAULT_HOST = 'limsdb2'
DEFAULT_PORT = 5432
DEFAULT_USERNAME = 'limsreader'


def psycopg2_select(query, database=DEFAULT_DATABASE, host=DEFAULT_HOST, port=DEFAULT_PORT, username=DEFAULT_USERNAME):

    connection = psycopg2.connect(
        'host={} port={} dbname={} user={}'.format(host, port, database, username), 
        cursor_factory=psycopg2.extras.RealDictCursor
    )
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        response = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    return response


def get_session_start_time(ecephys_session_id):
    session_info = psycopg2_select('select * from ecephys_sessions where id = {}'.format(ecephys_session_id))
    result = session_info[0]['date_of_acquisition']
    if result is None:
        return '1' # TODO: why no acquisition dates?
    return result


def get_probe_info(ecephys_session_id):
    return psycopg2_select('select * from ecephys_probes where ecephys_session_id = {}'.format(ecephys_session_id)) 
    


def main(ecephys_session_id, nwb_path, remove_file=False):

    source = 'Allen Institute for Brain Science'
    electrode_filtering = 'here is a description of our filtering' # TODO do we have a standard filtering string?

    session_start_time = get_session_start_time(ecephys_session_id)
    session_identifier = '{}'.format(ecephys_session_id)
    probe_info = get_probe_info(ecephys_session_id)

    api = EcephysLimsApi()

    # setup a file

    nwbfile = pynwb.NWBFile(
        source=source,
        session_description='EcephysSession',
        identifier=session_identifier,
        session_start_time=session_start_time,
        file_create_date=datetime.now()
    )


    # get a channel table and add nwb-required attributes
    channel_table = api.get_channel_table(ecephys_session_id)  # re: ids TODO: track these in lims and use globally valid ids - till then just use local
    channel_table['x'] = -1.0  # TODO when we get CCF positions from alignment we can write those here, till then there is no option to not supply these fields, so ...
    channel_table['y'] = -1.0
    channel_table['z'] = -1.0
    channel_table['imp'] = -1.0   # TODO: we don't currently have this info (at least not in a form I know about)
    channel_table['location'] = 'null'  # TODO: again, waits on CCF registration for accurate information. Will be acronym of CCF structure
    channel_table['filtering'] = electrode_filtering
    channel_table['group'] = None
    channel_table['group_name'] = ''

    # add probes (as devices), each with an electrode group

    for probe in probe_info:

        probe_nwb_device = Device(
            name=str(probe['id']), # why not name? probe names are actually codes for targeted structure. ids are the appropriate primary key
            source=source
        )

        probe_nwb_electrode_group = ElectrodeGroup(
            name=str(probe['id']),
            source=source, 
            description=probe['name'], # TODO probe name currently describes the targeting of the probe - the closest we have to a meaningful "kind"
            location='', # TODO not actailly sure where to get this
            device=probe_nwb_device
        )

        nwbfile.add_device(probe_nwb_device)
        nwbfile.add_electrode_group(probe_nwb_electrode_group)

        channel_table.loc[channel_table['probe_id'] == probe['id'], 'group'] = probe_nwb_electrode_group

    nwbfile.electrodes = ElectrodeTable().from_dataframe(channel_table, source=source, name='electrodes')

    if remove_file:
        os.remove(nwb_path)

    io = NWBHDF5IO(nwb_path, mode='w')
    io.write(nwbfile)
    io.close()

    return nwbfile, nwb_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ecephys_session_id', type=int) # 754312389, for instance
    parser.add_argument('--nwb_path', type=str, default=None)
    parser.add_argument('--remove_file', action='store_true', default=False)

    args = parser.parse_args()
    if args.nwb_path is None:
        nwb_path = '{}.nwb'.format(args.ecephys_session_id)
    else:
        nwb_path = args.nwb_path

    main(args.ecephys_session_id, nwb_path, args.remove_file)