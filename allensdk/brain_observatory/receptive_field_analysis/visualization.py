import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import matplotlib.gridspec as gridspec


def pvalue_to_NLL(p_values,
                  max_NLL=10.0):
    return np.where(p_values == 0.0, max_NLL, -np.log10(p_values))

def plot_receptive_field_data(receptive_field_data_dict, lsn, show=True, save_file_name=None, close=True):

    csid = receptive_field_data_dict['attrs']['csid']

    rts_on = receptive_field_data_dict['on']['rts']['data']
    rts_off = receptive_field_data_dict['off']['rts']['data']

    rts_on_blur = receptive_field_data_dict['on']['rts_convolution']['data']
    rts_off_blur = receptive_field_data_dict['off']['rts_convolution']['data']

    pvalues_on = receptive_field_data_dict['on']['pvalues']['data']
    pvalues_off = receptive_field_data_dict['off']['pvalues']['data']

    rf_on = pvalues_on.copy()
    rf_off = pvalues_off.copy()

    rf_on[np.logical_not(receptive_field_data_dict['on']['fdr_mask']['data'].sum(axis=0))] = np.nan
    rf_off[np.logical_not(receptive_field_data_dict['off']['fdr_mask']['data'].sum(axis=0))] = np.nan

    def plot_fields(on_data, off_data, on_axes, off_axes, clim=None):
        if clim is None:
            clim_max = max(on_data.max(), off_data.max())
            clim = (0,clim_max)
        on_axes.imshow(on_data, clim=clim, interpolation='none', origin='lower')
        img = off_axes.imshow(off_data, clim=clim, interpolation='none', origin='lower')
        cb = on_axes.figure.colorbar(img, ax=off_axes, ticks=clim)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        for frame in [on_axes, off_axes]:
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)

    # Prepare plotting figure:
    number_of_major_rows = 7
    fig = plt.figure(figsize=(4*1.9, number_of_major_rows*2))
    gs = gridspec.GridSpec(number_of_major_rows*2,4)
    ax_list = []

    # Plot chi-square summary:
    curr_axes = fig.add_subplot(gs[0:2, 1:3])
    ax_list += [curr_axes]
    chi_squared_grid = receptive_field_data_dict['chi_squared_analysis']['pvalues']['data']
    chi_square_grid_NLL = pvalue_to_NLL(chi_squared_grid)
    img = curr_axes.imshow(chi_square_grid_NLL, interpolation='none', origin='lower')

    cb = curr_axes.figure.colorbar(img, ax=curr_axes)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    curr_axes.axes.get_xaxis().set_visible(False)
    curr_axes.axes.get_yaxis().set_visible(False)
    curr_axes.set_title('Significant: %s (min_p=%s)' % (receptive_field_data_dict['chi_squared_analysis']['attrs']['significant'], receptive_field_data_dict['chi_squared_analysis']['attrs']['min_p']) )

    # MSR plot:
    curr_on_axes = fig.add_subplot(gs[2:4, 0:2])
    curr_off_axes = fig.add_subplot(gs[2:4, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    ind = lsn.data_set.get_cell_specimen_indices([csid])[0]
    min_clim = lsn.mean_response[:, :, ind,:].min()
    max_clim = lsn.mean_response[:, :, ind,:].max()
    plot_fields(lsn.mean_response[:, :, ind, 0], lsn.mean_response[:, :, ind, 1], curr_on_axes, curr_off_axes, clim=(min_clim, max_clim))

    # RTS no blur:
    curr_on_axes = fig.add_subplot(gs[4:6, 0:2])
    curr_off_axes = fig.add_subplot(gs[4:6, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_fields(rts_on, rts_off, curr_on_axes, curr_off_axes)

    # RTS no blur:
    curr_on_axes = fig.add_subplot(gs[6:8, 0:2])
    curr_off_axes = fig.add_subplot(gs[6:8, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_fields(rts_on_blur, rts_off_blur, curr_on_axes, curr_off_axes)

    # PValues:
    curr_on_axes = fig.add_subplot(gs[8:10, 0:2])
    curr_off_axes = fig.add_subplot(gs[8:10, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    clim_max = max(pvalues_on.max(), pvalues_off.max())
    plot_fields(pvalues_on, pvalues_off, curr_on_axes, curr_off_axes, clim=(0, clim_max/2))

    # Mask:
    curr_on_axes = fig.add_subplot(gs[10:12, 0:2])
    curr_off_axes = fig.add_subplot(gs[10:12, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_fields(rf_on, rf_off, curr_on_axes, curr_off_axes)


    # Gaussian fit:
    gf_on_exists = receptive_field_data_dict['on'].get('gaussian_fit',None)
    gf_off_exists = receptive_field_data_dict['off'].get('gaussian_fit', None)

    if gf_on_exists is None and gf_off_exists is None:
        pass
    elif (not gf_on_exists is None) and (not gf_off_exists is None):
        curr_on_axes = fig.add_subplot(gs[12:14, 0:2])
        curr_off_axes = fig.add_subplot(gs[12:14, 2:4])
        ax_list += [curr_on_axes, curr_off_axes]
        img_data_on = receptive_field_data_dict['on']['gaussian_fit']['data'].sum(axis=0)
        img_data_off = receptive_field_data_dict['off']['gaussian_fit']['data'].sum(axis=0)
        plot_fields(img_data_on, img_data_off, curr_on_axes, curr_off_axes)

    else:
        if not gf_on_exists is None:
            img_data = receptive_field_data_dict['on']['gaussian_fit']['data'].sum(axis=0)
            curr_axes = fig.add_subplot(gs[12:14, 0:2])
        else:
            img_data = receptive_field_data_dict['off']['gaussian_fit']['data'].sum(axis=0)
            curr_axes = fig.add_subplot(gs[12:14, 2:4])
        clim = (0, img_data.max())
        ax_list += [curr_axes]
        img = curr_axes.imshow(img_data, interpolation='none', origin='lower', clim=clim)
        cb = curr_axes.figure.colorbar(img, ax=curr_axes, ticks=clim)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        curr_axes.axes.get_xaxis().set_visible(False)
        curr_axes.axes.get_yaxis().set_visible(False)


    # gs.tight_layout(fig)

    plt.subplots_adjust(top=0.95)

    for ax in ax_list:
        ax.set_adjustable('box-forced')

    if not save_file_name is None:
        fig.savefig(save_file_name)

    if show == True:
        plt.show()

    if close:
        plt.close(fig)