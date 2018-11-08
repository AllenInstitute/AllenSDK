from __future__ import division
import multiprocessing as mp
import logging
import functools
import os

from six import iteritems
import SimpleITK as sitk
import numpy as np

from base_subimage import run_subimage
from classic_count_subimage import ClassicCountSubImage
from image_series_gridder import ImageSeriesGridder
from downsampling_utilities import downsample_average
from image_utilities import image_from_array, write_volume
from run_classic import ratio_and_pyramid, handle_pyramid

#==============================================================================


def run_count(grid_prefix, accumulator_prefix, target_spacings, **input_data):

    paths = []

    isg = ImageSeriesGridder(**input_data)
    isg.setup_subimages()

    isg.build_coarse_grids()

    cb = functools.partial(write_volume, name='sum_pixels', prefix=accumulator_prefix, paths=paths)
    isg.accumulator_to_numpy('sum_pixels', cb)

    ratio_and_pyramid(isg, 'sum_projecting_pixels',  'sum_pixels', 'projection_density',
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(isg, 'injection_sum_projecting_pixels',  'sum_pixels', 'injection_density',
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
