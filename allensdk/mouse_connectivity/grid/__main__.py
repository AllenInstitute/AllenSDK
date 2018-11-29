import logging
import pprint

import requests

from allensdk.core.multi_source_argschema_parser import MultiSourceArgschemaParser
from ._schemas import InputParameters, OutputParameters
from . import cases
from .image_series_gridder import ImageSeriesGridder


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

    try:
        case = cases[args['case']]
    except KeyError:
        logging.error('unrecognized case: {}'.format(args['case']))
        raise

    sub_images = args['sub_images']

    input_dimensions = [sub_images[0]['dimensions']['column'], 
                        sub_images[0]['dimensions']['row'], 
                        args['sub_image_count']]  

    input_spacing = [sub_images[0]['spacing']['column'], 
                     sub_images[0]['spacing']['row'], 
                     args['image_series_slice_spacing']] 

    for ii, si in enumerate(sub_images):
        del si['dimensions']
        del si['spacing']
        si['polygon_info'] = si['polygons']
        del si['polygons']
    sub_images = sorted(sub_images, key=lambda si: si['specimen_tissue_index'])

    output_dimensions = [args['reference_dimensions']['slice'], 
                         args['reference_dimensions']['row'], 
                         args['reference_dimensions']['column']]

    output_spacing = [args['reference_spacing']['slice'], 
                      args['reference_spacing']['row'], 
                      args['reference_spacing']['column']]

    subimage_kwargs = {'cls': case['subimage']}
    if args['filter_bit'] is not None:
        subimage_kwargs['filter_bit'] = args['filter_bit']

    gridder = ImageSeriesGridder(
        grid_prefix=input_data['grid_prefix'], 
        accumulator_prefix=input_data['accumulator_prefix'], 
        target_spacings=input_data['target_spacings'], 
        in_dims=input_dimensions, 
        in_spacing=input_spacing, 
        out_dims=output_dimensions, 
        out_spacing=output_spacing, 
        reduce_level=args.reduce_level, 
        subimages=sub_images, 
        subimage_kwargs=subimage_kwargs, 
        nprocesses=args.nprocesses, 
        affine_params=input_data['affine_params'], 
        dfmfld_path=input_data['deformation_field_path']
    )

    gridder.setup_subimages()
    gridder.build_coarse_grids()

    writer = case['writer']
    paths = writer(gridder)

    return {'output_file_paths': paths}


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