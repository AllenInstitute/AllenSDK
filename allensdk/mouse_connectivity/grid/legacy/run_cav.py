from __future__ import division
import multiprocessing as mp
import logging
import functools
import os

from six import iteritems
import SimpleITK as sitk
import numpy as np

from .base_subimage import run_subimage
from .image_series_gridder import ImageSeriesGridder
from .downsampling_utilities import downsample_average
from .image_utilities import image_from_array, write_volume


#==============================================================================

    
def run_cav(grid_prefix, accumulator_prefix, **input_data):

    paths = []

    isg = ImageSeriesGridder(**input_data)
    isg.setup_subimages()

    isg.build_coarse_grids()

    cb = functools.partial(write_volume, name='cav_tracer_10', prefix=accumulator_prefix, paths=paths)
    isg.accumulator_to_numpy('cav_tracer', cb)

    cb = functools.partial(write_volume, name='sum_pixels_10', prefix=accumulator_prefix, paths=paths)
    isg.accumulator_to_numpy('sum_pixels', cb)

    isg.make_ratio_volume('cav_tracer', 'sum_pixels', 'cav_density')
    isg.volumes['cav_density'] = image_from_array(isg.volumes['cav_density'], isg.out_spacing)
    write_volume(isg.volumes['cav_density'], name='cav_density_10', prefix=grid_prefix, paths=paths)
    del isg.volumes['cav_density']

    isg.volumes['data_mask'] = isg.volumes['sum_pixels'] / np.amax(isg.volumes['sum_pixels'])
    del isg.volumes['sum_pixels']
    isg.volumes['data_mask'][isg.volumes['data_mask'] > 0.5] = 1
    isg.volumes['data_mask'][isg.volumes['data_mask'] < 0.5] = 0
    isg.volumes['data_mask'] = image_from_array(isg.volumes['data_mask'], isg.out_spacing)
    write_volume(isg.volumes['data_mask'], name='data_mask_10', prefix=accumulator_prefix, paths=paths)
    del isg.volumes

    return paths


#==============================================================================





































