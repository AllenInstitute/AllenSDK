import logging
import pprint

import requests

from allensdk.core.multi_source_argschema_parser import MultiSourceArgschemaParser
from ._schemas import InputParameters, OutputParameters

def get_inputs_from_lims(host, image_series_id, output_root, job_queue, strategy):
    
    uri = ''.join('''
        {}/input_jsons?
        object_id={}&
        object_class=ImageSeries&
        strategy_class={}&
        job_queue_name={}
    '''.format(host, image_series_id, strategy, job_queue).split())
    response = requests.get(uri)
    data = response.json()

    if len(data) == 1 and 'error' in data:
        raise ValueError('bad request uri: {} ({})'.format(uri, data['error']))

    return data
    

def run_grid(args):

    

    return {}


def main():

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    parser = MultiSourceArgschemaParser(
        sources={
            'lims': {
                'get_input_data': get_inputs_from_lims,
                'params': {
                    'host': 'http://lims2',
                    'job_queue': None,
                    'strategy': None,
                    'image_series_id': None,
                    'output_root': None
                }
            }
        },
        schema_type=InputParameters,
        output_schema_type=OutputParameters,
    )

    output = run_grid(parser.args)
    MultiSourceArgschemaParser.write_or_print_outputs(output, parser)


if __name__ == '__main__':
    main()