from __future__ import division
import logging
import sys
import functools

import numpy as np
from scipy.ndimage.interpolation import zoom
from six import iteritems

from allensdk.mouse_connectivity.grid.utilities import image_utilities as iu
from .base_subimage import PolygonSubImage, SegmentationSubImage, IntensitySubImage


#==============================================================================


class CavSubImage(PolygonSubImage):

    required_polys = ['missing_tile', 'cav_tracer']


    def compute_coarse_planes(self):

        nonmissing = np.logical_not(self.images['missing_tile'])
        del self.images['missing_tile']

        self.apply_pixel_counter('sum_pixels', nonmissing)

        cav_nonmissing = np.multiply(self.images['cav_tracer'], nonmissing)
        del nonmissing
        self.apply_pixel_counter('cav_tracer', cav_nonmissing)
        del cav_nonmissing

        del self.images


#==============================================================================
