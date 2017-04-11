from .core import get_receptive_field_data_dict_with_postprocessing
import numpy as np
from  .tools import dict_generator
from .utilities import get_attribute_dict
import pandas as pd

def receptive_field_analysis(lsn):

    metadata = lsn.data_set.get_metadata()

    oeid = metadata['ophys_experiment_id']
    csid_list = lsn.data_set.get_cell_specimen_ids()

    # receptive_field_analysis_data = {}
    # for key in ['receptive_field', 'pvalues', 'rts', 'rts_convolution', 'fdr_mask', 'fdr_corrected']:
    #     receptive_field_analysis_data[key] = np.zeros((lsn.nrows, lsn.ncols, lsn.numbercells + 1, 2))
    # receptive_field_analysis_data['chi_squared_analysis'] = np.zeros((lsn.nrows, lsn.ncols, lsn.numbercells + 1))
    # receptive_field_analysis_data['gaussian_fit'] = np.zeros((int(np.ceil(lsn.nrows*4.65)), int(np.ceil(lsn.ncols*4.65)), lsn.numbercells + 1,2))

    # df_list = []
    csid_receptive_field_data_dict = {}
    import warnings
    warnings.warn('TAKE AWAY THIS 2')
    for csid in csid_list[:2]:

        cell_index = lsn.data_set.get_cell_specimen_indices(cell_specimen_ids=[csid])[0]
        curr_receptive_field_data_dict = get_receptive_field_data_dict_with_postprocessing(lsn.data_set, csid, alpha=.05)
        csid_receptive_field_data_dict[str(csid)] = curr_receptive_field_data_dict

    return csid_receptive_field_data_dict


        # import warnings
        # warnings.warn("DEBUG __init__.py")
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(receptive_field_data_dict['on']['rts_convolution']['data'], interpolation='none', origin='lower')
        # ax[1].imshow(receptive_field_data_dict['off']['rts_convolution']['data'], interpolation='none', origin='lower')
        # plt.show()


    #     # Build attribute dataframe:
    #     curr_attribute_dict = get_attribute_dict(receptive_field_data_dict)
    #     massaged_dict = {}
    #     for key, val in curr_attribute_dict.items():
    #         massaged_dict[key] = [val]
    #     curr_df = pd.DataFrame.from_dict(massaged_dict)
    #     df_list.append(curr_df)
    #
    #     # Receptive field:
    #     rf_on = receptive_field_data_dict['on']['rts_convolution']['data'].copy()
    #     rf_off = receptive_field_data_dict['off']['rts_convolution']['data'].copy()
    #     rf_on[np.logical_not(receptive_field_data_dict['on']['fdr_mask']['data'].sum(axis=0))] = np.nan
    #     rf_off[np.logical_not(receptive_field_data_dict['off']['fdr_mask']['data'].sum(axis=0))] = np.nan
    #     receptive_field_analysis_data['receptive_field'][:,:,cell_index, 0] = rf_on
    #     receptive_field_analysis_data['receptive_field'][:, :, cell_index, 1] = rf_off
    #
    #     for on_off_key, on_off_ind in zip(['on', 'off'],[0,1]):
    #         for key in ['pvalues', 'rts', 'rts_convolution', 'fdr_mask', 'fdr_corrected']:
    #             print key
    #             receptive_field_analysis_data[key][:, :, cell_index, on_off_ind] = receptive_field_data_dict[on_off_key][key]['data']
    #
    #     for on_off_key, on_off_ind in zip(['on', 'off'], [0, 1]):
    #         if 'gaussian_fit' in receptive_field_data_dict[on_off_key]:
    #             receptive_field_analysis_data['gaussian_fit'][:, :, cell_index, on_off_ind] = receptive_field_data_dict[on_off_key]['gaussian_fit']['data']
    #
    # metadata_df = pd.concat(df_list)
    # return receptive_field_analysis_data, metadata_df

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(rf_on, interpolation='none', origin='lower')
# ax[1].imshow(rf_off, interpolation='none', origin='lower')
# plt.show()

# Respone triggered stimulus


    #
    # store = pd.HDFStore('trial.h5', mode='a')
    #
    # for k, v in [('metadata_df', metadata_df)]:
    #     store.put('analysis/%s' % (k), v)
    #
    # store.close()

# import warnings
# warnings.warn('Plotting fore debugging')
# print receptive_fields.shape
# print rf_on.shape, rf_off.shape
# from allensdk.core.brain_observatory_cache import BrainObservatoryCache
# import receptivefield
# receptivefield.brain_observatory_cache = BrainObservatoryCache(manifest_file='/data/mat/nicholasc/boc_iwarehouse/manifest.json')
# from receptivefield.core.visualization import plot_receptive_field_data
# plot_receptive_field_data(receptive_field_data_dict)
# break
