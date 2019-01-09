import logging
import sys
import argparse
from datetime import datetime

import argschema
import pynwb

from ._schemas import InputSchema, OutputSchema


def get_inputs_from_lims(host, ecephys_session_id, output_root, job_queue, strategy):
    
    uri = ''.join('''
        {}/input_jsons?
        object_id={}&
        object_class=EcephysSession&
        strategy_class={}&
        job_queue_name={}
    '''.format(host, ecephys_session_id, strategy, job_queue).split())
    response = requests.get(uri)
    data = response.json()

    if len(data) == 1 and 'error' in data:
        raise ValueError('bad request uri: {} ({})'.format(uri, data['error']))

    data['output_path'] = os.path.join(output_root, os.path.split(data['output_path'])[-1])
    return data


def write_or_print_outputs(data, parser):
    data.update({'input_parameters': parser.args})
    if 'output_json' in parser.args:
        parser.output(data, indent=2)
    else:
        print(parser.get_output_json(data))   


def write_ecephys_nwb(output_path, session_id, session_start_time, **kwargs):


    nwbfile = pynwb.NWBFile(
        session_description='EcephysSession',
        identifier='{}'.format(session_id),
        session_start_time=session_start_time,
        file_create_date=datetime.now()
    )

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


    parser = argschema.ArgSchemaParser(
        args=remaining_args,
        input_data=input_data,
        schema_type=InputSchema,
        output_schema_type=OutputSchema,
    )

    output = write_ecephys_nwb(**parser.args)
    write_or_print_outputs(output, parser)


if __name__ == '__main__':
    main()
