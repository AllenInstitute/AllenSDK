import argparse
from datetime import datetime

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np

import pynwb
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup


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


def main(ecephys_session_id, nwb_path):

    source = 'Allen Institute for Brain Science'
    electrode_filtering = 'here is a description of our filtering' # TODO do we have a standard filtering string?

    session_start_time = get_session_start_time(ecephys_session_id)
    session_identifier = '{}'.format(ecephys_session_id)
    probe_info = get_probe_info(ecephys_session_id)

    # setup a file

    nwbfile = pynwb.NWBFile(
        source=source,
        session_description='EcephysSession',
        identifier=session_identifier,
        session_start_time=session_start_time,
        file_create_date=datetime.now()
    )


    # add columns to the electrode table
    nwbfile.add_electrode_column(name='local_channel_index', description='an index into a flattened array of all of the channels on this probe')
    nwbfile.add_electrode_column(name='mask', description='true if this channel\'s data can be used, false otherwise (if it is damaged, or a reference channel, for instance)')
    nwbfile.add_electrode_column(name='vertical_pos', description='position along the length of the probe in microns. Higher is deeper.')
    nwbfile.add_electrode_column(name='horizontal_pos', description='position along the width of the probe, in microns')

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

        max_vertical_pos = np.amax(probe['probe_info']['vertical_pos'])
        for ii, local_index in enumerate(probe['probe_info']['channel']):
            nwbfile.add_electrode(
                x=-1.0, # TODO when we get CCF positions from alignment we can write those here, till then there is no option to not supply these fields, so ...
                y=-1.0,
                z=-1.0,
                imp=-1.0, # TODO: we don't currently have this info (at least not in a form I know about)
                location='null', # TODO: again, waits on CCF registration for accurate information. Will be acronym of CCF structure
                filtering=electrode_filtering,
                group=probe_nwb_electrode_group,
                group_name=None, # use group?
                local_channel_index = local_index,
                mask=probe['probe_info']['mask'][ii],
                vertical_pos=probe['probe_info']['vertical_pos'][ii] - max_vertical_pos, 
                horizontal_pos=probe['probe_info']['vertical_pos'][ii]
                # id= TODO: track these in lims and use globally valid ids - till then just use local
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ecephys_session_id', type=int) # 754312389, for instance
    parser.add_argument('--nwb_path', type=str, default=None)

    args = parser.parse_args()
    if args.nwb_path is None:
        nwb_path = '{}.nwb'.format(args.ecephys_session_id)
    else:
        nwb_path = args.nwb_path

    main(args.ecephys_session_id, nwb_path)