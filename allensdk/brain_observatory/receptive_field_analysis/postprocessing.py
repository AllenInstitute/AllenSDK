from .fit_parameters import get_gaussian_fit_single_channel, compute_distance, compute_overlap
from .chisquarerf import chi_square_binary, get_peak_significance, pvalue_to_NLL
from .utilities import upsample_image_to_degrees
import collections
import numpy as np
import sys

def get_gaussian_fit(receptive_field_data_dict):

    fit_parameters_dict_combined = {'on':collections.defaultdict(list), 'off':collections.defaultdict(list)}
    counter = {'on':0, 'off':0}
    for on_off_key in ['on', 'off']:
        fit_parameters_dict = fit_parameters_dict_combined[on_off_key]
        for ci in range(receptive_field_data_dict[on_off_key]['fdr_mask']['attrs']['number_of_components']):
            curr_component_mask = upsample_image_to_degrees(np.logical_not(receptive_field_data_dict[on_off_key]['fdr_mask']['data'][ci,:,:])) > .5
            rf_response = upsample_image_to_degrees(receptive_field_data_dict[on_off_key]['rts_convolution']['data'].copy())
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

def run_postprocessing(data, receptive_field_data_dict):

    stimulus = receptive_field_data_dict['attrs']['stimulus']

    # Gaussian fit postprocessing:
    fit_parameters_dict_combined, counter = get_gaussian_fit(receptive_field_data_dict)

    for on_off_key in ['on', 'off']:

        if counter[on_off_key] > 0:

            receptive_field_data_dict[on_off_key]['gaussian_fit'] = {}
            receptive_field_data_dict[on_off_key]['gaussian_fit']['attrs'] = {}

            fit_parameters_dict = fit_parameters_dict_combined[on_off_key]
            for key, val in fit_parameters_dict.items():

                if key == 'data':
                    receptive_field_data_dict[on_off_key]['gaussian_fit']['data'] = np.array(val)
                else:
                    receptive_field_data_dict[on_off_key]['gaussian_fit']['attrs'][key] = np.array(val)

    # Chi squared test statistic postprocessing:
    cell_index = receptive_field_data_dict['attrs']['cell_index']
    locally_sparse_noise_template = data.get_stimulus_template(stimulus)

    event_array = np.zeros((receptive_field_data_dict['event_vector']['data'].shape[0], 1), dtype=np.bool)
    event_array[:,0] = receptive_field_data_dict['event_vector']['data']

    chi_squared_grid = chi_square_binary(event_array, locally_sparse_noise_template)
    alpha = receptive_field_data_dict['on']['fdr_mask']['attrs']['alpha']
    assert receptive_field_data_dict['off']['fdr_mask']['attrs']['alpha'] == alpha
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

    receptive_field_data_dict['chi_squared_analysis'] = chi_squared_grid_dict

    return receptive_field_data_dict

if __name__ == "__main__":

    from receptivefield.core import run_receptive_field_computation, print_summary, write_receptive_field_data_dict_to_h5, read_receptive_field_data_dict_from_h5
    from receptivefield.core.visualization import plot_receptive_field_data

    # csid = 517472416 # triple!
    csid = 517526760 # two ON
    # csid = 539917553
    # csid = 540988186


    receptive_field_data_dict = run_receptive_field_computation(csid, alpha=.05)
    run_postprocessing(receptive_field_data_dict)


    # write_receptive_field_data_dict_to_h5(receptive_field_data_dict, 'tmp.h5')
    # receptive_field_data_dict = read_receptive_field_data_dict_from_h5('tmp.h5')

    print_summary(receptive_field_data_dict)

    plot_receptive_field_data(receptive_field_data_dict)

    # sys.exit()


    # rf_df, param_dict = get_gaussian_fit(receptive_field_data_dict)
    #
    # rf_response_triggered_blur_on_blur = receptive_field_data_dict['on']['rts_convolution']['data'].copy()
    # rf_response_triggered_blur_off_blur = receptive_field_data_dict['off']['rts_convolution']['data'].copy()
    #
    # rf_response_triggered_blur_on_blur[np.logical_not(receptive_field_data_dict['on']['fdr_mask']['data'])] = 0
    # rf_response_triggered_blur_off_blur[np.logical_not(receptive_field_data_dict['off']['fdr_mask']['data'])] = 0
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(rf_response_triggered_blur_on_blur, interpolation='none')
    # ax[0,1].imshow(rf_response_triggered_blur_off_blur, interpolation='none')
    #
    # if not param_dict[0]['data_fitted_on'] is None:
    #     ax[1, 0].imshow(param_dict[0]['data_fitted_on'][0], interpolation='none')
    # else:
    #     ax[1, 0].set_visible(False)
    #
    # if not param_dict[0]['data_fitted_off'] is None:
    #     ax[1, 1].imshow(param_dict[0]['data_fitted_off'][0], interpolation='none')
    # else:
    #     ax[1, 1].set_visible(False)
    #
    # plt.show()



    # print pvalues_on.shape
    # print pvalues_off.shape

    # p_values =
    # get_gaussian_fit        if not param_dict[0]['data_fitted_%s' % on_off_key] is None:
    #         curr_attrs_dict = {}
    #         assert len(param_dict[0]['data_fitted_%s' % on_off_key][1]) == 6
    #         for val, key in zip(param_dict[0]['data_fitted_%s' % on_off_key][1], ['height', 'center_y', 'center_x', 'width_y', 'width_x', 'rotation']):
    #             curr_attrs_dict[key] = val
    #         receptive_field_data_dict[on_off_key]['gaussian_fit'] = {'data': param_dict[0]['data_fitted_%s' % on_off_key][0], 'attrs':curr_attrs_dict}