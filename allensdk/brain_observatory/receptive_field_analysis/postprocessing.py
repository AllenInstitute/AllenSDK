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

from .fit_parameters import get_gaussian_fit_single_channel, compute_distance, compute_overlap
from .chisquarerf import chi_square_binary, get_peak_significance, pvalue_to_NLL
from .utilities import upsample_image_to_degrees
import collections
import numpy as np
import sys

def get_gaussian_fit(rf):

    fit_parameters_dict_combined = {'on':collections.defaultdict(list), 'off':collections.defaultdict(list)}
    counter = {'on':0, 'off':0}
    for on_off_key in ['on', 'off']:
        fit_parameters_dict = fit_parameters_dict_combined[on_off_key]
        for ci in range(rf[on_off_key]['fdr_mask']['attrs']['number_of_components']):
            curr_component_mask = upsample_image_to_degrees(np.logical_not(rf[on_off_key]['fdr_mask']['data'][ci,:,:])) > .5
            rf_response = upsample_image_to_degrees(rf[on_off_key]['rts_convolution']['data'].copy())
            rf_response[curr_component_mask] = 0

            if rf_response.sum() > 0:
                get_gaussian_fit_single_channel(rf_response, fit_parameters_dict)
                counter[on_off_key] += 1

    for ii_off in range(counter['on']):
        fit_parameters_dict_combined['on']['distance'].append([None]*counter['off'])
        fit_parameters_dict_combined['on']['overlap'].append([None] * counter['off'])

    for ii_off in range(counter['off']):
        fit_parameters_dict_combined['off']['distance'].append([None] * counter['on'])
        fit_parameters_dict_combined['off']['overlap'].append([None]*counter['on'])


    for ii_on in range(counter['on']):
        for ii_off in range(counter['off']):
            center_on = fit_parameters_dict_combined['on']['center_x'][ii_on], fit_parameters_dict_combined['on']['center_y'][ii_on]
            center_off = fit_parameters_dict_combined['off']['center_x'][ii_off], fit_parameters_dict_combined['off']['center_y'][ii_off]
            curr_distance = compute_distance(center_on, center_off)
            fit_parameters_dict_combined['on']['distance'][ii_on][ii_off] = curr_distance
            fit_parameters_dict_combined['off']['distance'][ii_off][ii_on] = curr_distance

            data_on = fit_parameters_dict_combined['on']['data'][ii_on]
            data_off = fit_parameters_dict_combined['off']['data'][ii_off]
            curr_overlap = compute_overlap(data_on, data_off)
            fit_parameters_dict_combined['on']['overlap'][ii_on][ii_off] = curr_overlap
            fit_parameters_dict_combined['off']['overlap'][ii_off][ii_on] = curr_overlap

    return fit_parameters_dict_combined, counter

def run_postprocessing(data, rf):

    stimulus = rf['attrs']['stimulus']

    # Gaussian fit postprocessing:
    fit_parameters_dict_combined, counter = get_gaussian_fit(rf)

    for on_off_key in ['on', 'off']:

        if counter[on_off_key] > 0:

            rf[on_off_key]['gaussian_fit'] = {}
            rf[on_off_key]['gaussian_fit']['attrs'] = {}

            fit_parameters_dict = fit_parameters_dict_combined[on_off_key]
            for key, val in fit_parameters_dict.items():

                if key == 'data':
                    rf[on_off_key]['gaussian_fit']['data'] = np.array(val)
                else:
                    rf[on_off_key]['gaussian_fit']['attrs'][key] = np.array(val)

    # Chi squared test statistic postprocessing:
    cell_index = rf['attrs']['cell_index']
    locally_sparse_noise_template = data.get_stimulus_template(stimulus)

    event_array = np.zeros((rf['event_vector']['data'].shape[0], 1), dtype=np.bool)
    event_array[:,0] = rf['event_vector']['data']

    chi_squared_grid = chi_square_binary(event_array, locally_sparse_noise_template)
    alpha = rf['on']['fdr_mask']['attrs']['alpha']
    assert rf['off']['fdr_mask']['attrs']['alpha'] == alpha
    chi_square_grid_NLL = pvalue_to_NLL(chi_squared_grid)

    peak_significance = get_peak_significance(chi_square_grid_NLL, locally_sparse_noise_template,  alpha=alpha)
    significant = peak_significance[0][0]
    min_p = peak_significance[1][0]
    pvalues_chi_square = peak_significance[2][0]
    best_exclusion_region_mask = peak_significance[3][0]

    chi_squared_grid_dict = {
                             'best_exclusion_region_mask':{'data':best_exclusion_region_mask},
                             'attrs':{'significant':significant, 'alpha': alpha, 'min_p':min_p},
                             'pvalues':{'data':pvalues_chi_square}
                             }

    rf['chi_squared_analysis'] = chi_squared_grid_dict

    return rf

if __name__ == "__main__":
    # csid = 517472416 # triple!
    csid = 517526760 # two ON
    # csid = 539917553
    # csid = 540988186
