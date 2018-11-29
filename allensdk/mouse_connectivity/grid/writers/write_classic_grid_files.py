import functools

import numpy as np

from ..image_utilities import write_volume
from ..downsampling_utilities import downsample_average

def write_classic_grid_files(grid_directory, accumulator_directory, target_spacings, gridder):

    cb = functools.partial(write_volume, name='sum_pixel_intensities', prefix=accumulator_prefix, paths=paths)
    gridder.consume_volume('sum_pixel_intensities', cb)

    cb = functools.partial(write_volume, name='injection_sum_pixel_intensities', prefix=accumulator_prefix, paths=paths)
    gridder.consume_volume('injection_sum_pixel_intensities', cb)

    cb = functools.partial(write_volume, name='sum_pixels', prefix=accumulator_prefix, paths=paths)
    gridder.accumulator_to_numpy('sum_pixels', cb)

    ratio_and_pyramid(gridder, 'sum_projecting_pixels',  'sum_pixels', 'projection_density', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(gridder, 'injection_sum_projecting_pixels',  'sum_pixels', 'injection_density', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(gridder, 'sum_projecting_pixel_intensities',  'sum_pixels', 'projection_energy', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(gridder, 'injectionsum_projecting_pixel_intensities',  'sum_pixels', 'injection_energy', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(gridder, 'injection_sum_pixels',  'sum_pixels', 'injection_fraction', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    ratio_and_pyramid(gridder, 'aav_exclusion_sum_pixels',  'sum_pixels', 'aav_exclusion_fraction', 
                      accumulator_prefix, grid_prefix, target_spacings, paths=paths)

    gridder.volumes['data_mask'] = gridder.volumes['sum_pixels'] / np.amax(gridder.volumes['sum_pixels'])
    del gridder.volumes['sum_pixels']
    gridder.volumes['data_mask'][gridder.volumes['data_mask'] > 0.5] = 1
    gridder.volumes['data_mask'][gridder.volumes['data_mask'] < 0.5] = 0
    handle_pyramid(gridder, 'data_mask', target_spacings, grid_prefix, paths=paths)

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