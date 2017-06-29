# Copyright 2017 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

from .fitgaussian2D import fitgaussian2D, GaussianFitError, gaussian2D
import numpy as np
import pandas as pd
import collections
import sys
import warnings

def add_to_fit_parameters_dict_single(fit_parameters_dict, p):

    fit_parameters_dict['height'].append(p[0])
    fit_parameters_dict['center_y'].append(p[1])
    fit_parameters_dict['center_x'].append(p[2])
    fit_parameters_dict['width_y'].append(p[3])
    fit_parameters_dict['width_x'].append(p[4])
    fit_parameters_dict['rotation'].append(p[5])
    if (p[3] is None) or (p[4] is None):
        fit_parameters_dict['area'].append(None)
    else:
        fit_parameters_dict['area'].append(np.pi * (3./2) ** 2 * np.abs(p[3]) * np.abs(p[4]))

def get_gaussian_fit_single_channel(rf, fit_parameters_dict):

    try:
        p_fit = fitgaussian2D(rf)
        add_to_fit_parameters_dict_single(fit_parameters_dict, p_fit)
        data_fitted_on = gaussian2D(*p_fit)(*np.indices(rf.shape))
        fit_parameters_dict['data'].append(data_fitted_on)
    except GaussianFitError:
        warnings.warn('GaussianFitError (on subfield) caught')
        add_to_fit_parameters_dict_single(fit_parameters_dict, [None]*6)
        fit_parameters_dict['data'].append(np.zeros_like(rf))

def compute_distance(center_on, center_off):

    center_x_on, center_y_on = center_on
    center_x_off, center_y_off = center_off

    if (center_x_on is None) or (center_y_on is None) or (center_x_off is None) or (center_y_off is None):
        return None
    else:
        return np.sqrt((center_x_off-center_x_on)**2+(center_y_off-center_y_on)**2)

def compute_overlap(data_fitted_on, data_fitted_off):

    on_bin = np.where(data_fitted_on > 0.001, 1, 0)
    off_bin = np.where(data_fitted_off > 0.001, 1, 0)

    return float((np.multiply(on_bin, off_bin)).sum()) / (np.sqrt(on_bin.sum()) * np.sqrt(off_bin.sum()))



