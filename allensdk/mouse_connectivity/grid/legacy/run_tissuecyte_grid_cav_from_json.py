import argparse
import json
import logging

import numpy as np
import nrrd

from allensdk.internal.core.lims_pipeline_module import PipelineModule, run_module
from allensdk.internal.mouse_connectivity.tissuecyte_gridding.run_cav import run_cav
from allensdk.internal.mouse_connectivity.tissuecyte_gridding.cav_subimage import CavSubImage


def main():

    mod = PipelineModule()
    mod.parser.add_argument("--nprocesses", type=int, default=1)
    mod.parser.add_argument("--reduce_level", type=int, default=0)
    mod.parser.add_argument("--filter_bit", type=int, default=None)
    args = mod.parser.parse_args()

    logging.info("reading input")
    input_data = mod.input_data()

    sub_images = input_data['sub_images']

    input_dimensions = [sub_images[0]['dimensions']['column'], 
                        sub_images[0]['dimensions']['row'], 
                        input_data['sub_image_count']]  

    input_spacing = [sub_images[0]['spacing']['column'], 
                     sub_images[0]['spacing']['row'], 
                     input_data['image_series_slice_spacing']] 


    for ii, si in enumerate(sub_images):
        del si['dimensions']
        del si['spacing']
        si['polygon_info'] = si['polygons']
        del si['polygons']
    sub_images = sorted(sub_images, key=lambda si: si['specimen_tissue_index'])

    output_dimensions = [input_data['reference_dimensions']['slice'], 
                         input_data['reference_dimensions']['row'], 
                         input_data['reference_dimensions']['column']]

    output_spacing = [input_data['reference_spacing']['slice'], 
                      input_data['reference_spacing']['row'], 
                      input_data['reference_spacing']['column']]

    subimage_kwargs = {'cls': CavSubImage}
    if args.filter_bit is not None:
        subimage_kwargs['filter_bit'] = args.filter_bit

    paths = run_cav(grid_prefix=input_data['grid_prefix'], 
                    accumulator_prefix=input_data['accumulator_prefix'], 
                    in_dims=input_dimensions, 
                    in_spacing=input_spacing, 
                    out_dims=output_dimensions, 
                    out_spacing=output_spacing, 
                    reduce_level=args.reduce_level, 
                    subimages=sub_images, 
                    subimage_kwargs=subimage_kwargs, 
                    nprocesses=args.nprocesses, 
                    affine_params=input_data['affine_params'], 
                    dfmfld_path=input_data['deformation_field_path'])
            

    logging.info('writing output')
    mod.write_output_data({'output_file_paths': paths})


if __name__ == '__main__':
    
#    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('').setLevel(logging.INFO)
    
    main()
