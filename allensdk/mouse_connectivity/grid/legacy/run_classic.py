from __future__ import division
import multiprocessing as mp
import logging
import functools
import os

from six import iteritems
import SimpleITK as sitk
import numpy as np

from .base_subimage import run_subimage
from .classic_subimage import ClassicSubImage
from .image_series_gridder import ImageSeriesGridder
from .downsampling_utilities import downsample_average
from .image_utilities import image_from_array, write_volume


#==============================================================================

    
def run_classic(grid_prefix, accumulator_prefix, target_spacings, **input_data):

    paths = []

    isg = ImageSeriesGridder(**input_data)
    isg.setup_subimages()

    isg.build_coarse_grids()

    cb = functools.partial(write_volume, name='sum_pixel_intensities', prefix=accumulator_prefix, paths=paths)
    isg.consume_volume('sum_pixel_intensities', cb)

    cb = functools.partial(write_volume, name='injection_sum_pixel_intensities', prefix=accumulator_prefix, paths=paths)
    isg.consume_volume('injection_sum_pixel_intensities', cb)

    cb = functools.partial(write_volume, name='sum_pixels', prefix=accumulator_prefix, paths=paths)
    isg.accumulator_to_numpy('sum_pixels', cb)

    ratio_and_pyramid(isg, 'sum_projecting_pixels',  'sum_pixels', 'projection_density', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(isg, 'injection_sum_projecting_pixels',  'sum_pixels', 'injection_density', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(isg, 'sum_projecting_pixel_intensities',  'sum_pixels', 'projection_energy', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(isg, 'injectionsum_projecting_pixel_intensities',  'sum_pixels', 'injection_energy', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(isg, 'injection_sum_pixels',  'sum_pixels', 'injection_fraction', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(isg, 'aav_exclusion_sum_pixels',  'sum_pixels', 'aav_exclusion_fraction', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    isg.volumes['data_mask'] = isg.volumes['sum_pixels'] / np.amax(isg.volumes['sum_pixels'])
    del isg.volumes['sum_pixels']
    isg.volumes['data_mask'][isg.volumes['data_mask'] > 0.5] = 1
    isg.volumes['data_mask'][isg.volumes['data_mask'] < 0.5] = 0
    handle_pyramid(isg, 'data_mask', target_spacings, grid_prefix, paths=paths)

    return paths


def handle_pyramid(isg, key, target_spacings, prefix, paths):

    cspacing = isg.out_spacing[0]
    for tspacing in target_spacings:

        downsampled = downsample_average(isg.volumes[key], cspacing, tspacing)
        write_volume(image_from_array(downsampled, [tspacing] * 3), 
                     key, prefix=prefix, specify_resolution=tspacing, paths=paths)

    del isg.volumes[key]


def ratio_and_pyramid(isg, num, den, out, accumulator_prefix, grid_prefix, target_spacings, paths):

    cb = functools.partial(write_volume, name=num, prefix=accumulator_prefix, paths=paths)
    isg.accumulator_to_numpy(num, cb)
        
    isg.make_ratio_volume(num, den, out)
    del isg.volumes[num]
    handle_pyramid(isg, out, target_spacings, grid_prefix, paths)


#==============================================================================





































