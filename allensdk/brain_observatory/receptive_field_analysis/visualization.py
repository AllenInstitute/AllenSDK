import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEFAULT_CMAP = 'viridis'

def plot_ellipses(gaussian_fit_dict, ax=None, show=True, close=True, save_file_name=None, color='b'):
    '''Example Usage:
    oeid, cell_index, stimulus = 512176430, 12, 'locally_sparse_noise'
    brain_observatory_cache = BrainObservatoryCache()
    data_set = brain_observatory_cache.get_ophys_experiment_data(oeid)
    lsn = LocallySparseNoise(data_set, stimulus)
    result = get_receptive_field_data_dict_with_postprocessing(data_set, cell_index, stimulus, alpha=.05, number_of_shuffles=5000)
    plot_ellipses(result['off']['gaussian_fit'], color='r')
    '''

    if ax is None:
        fig, ax = plt.subplots(1)
        ax.set_xlim(0, 130)
        ax.set_ylim(0, 74)
        plt.axis('off')

    # try:
    on_comp = len(gaussian_fit_dict['attrs']['center_x'])
    for i in range(on_comp):
        xy = (gaussian_fit_dict['attrs']['center_x'][i], gaussian_fit_dict['attrs']['center_y'][i])
        width = 3 * np.abs(gaussian_fit_dict['attrs']['width_x'][i])
        height = 3 * np.abs(gaussian_fit_dict['attrs']['width_y'][i])
        angle = gaussian_fit_dict['attrs']['rotation'][i]
        if np.logical_not(any(np.isnan(xy))):
            ellipse = mpatches.Ellipse(xy, width=width, height=height, angle=angle, lw=2, edgecolor=color,
                                       facecolor='none')
            ax.add_artist(ellipse)

    if not save_file_name is None:
        fig.savefig(save_file_name)

    if show == True:
        plt.show()

    if close:
        plt.close(fig)

    # except:
    #     pass

    return ax

def pvalue_to_NLL(p_values,
                  max_NLL=10.0):
    return np.where(p_values == 0.0, max_NLL, -np.log10(p_values))

def plot_chi_square_summary(rf_data, ax=None, cmap=DEFAULT_CMAP):
    if ax is None:
        ax = plt.gca()

    chi_squared_grid = rf_data['chi_squared_analysis']['pvalues']['data']
    chi_square_grid_NLL = pvalue_to_NLL(chi_squared_grid)
    clim = (0, max(2,chi_square_grid_NLL.max()))
    img = ax.imshow(chi_square_grid_NLL, interpolation='none', origin='lower', clim=clim, cmap=cmap)

    cb = ax.figure.colorbar(img, ax=ax, ticks=clim, fraction=0.024, pad=0.04)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title('Significant: %s (min_p=%s)' % (rf_data['chi_squared_analysis']['attrs']['significant'], 
                                                 rf_data['chi_squared_analysis']['attrs']['min_p']) )    

def plot_msr_summary(lsn, cell_index, ax_on, ax_off):
    ax_list += [curr_on_axes, curr_off_axes]
    min_clim = lsn.mean_response[:, :, cell_index,:].min()
    max_clim = lsn.mean_response[:, :, cell_index,:].max()
    plot_fields(lsn.mean_response[:, :, cell_index, 0], 
                lsn.mean_response[:, :, cell_index, 1], 
                curr_on_axes, curr_off_axes, clim=(min_clim, max_clim))

def plot_fields(on_data, off_data, on_axes, off_axes, clim=None, cmap=DEFAULT_CMAP):
    on_axes.figure.subplots_adjust(right=0.9)
    cbar_axes = on_axes.figure.add_axes([0.93, 0.37, 0.02, .28])
    
    if clim is None:
        clim_max = max(on_data.max(), off_data.max())
        clim = (0,clim_max)
    on_axes.imshow(on_data, clim=clim, cmap=cmap, interpolation='none', origin='lower')
    img = off_axes.imshow(off_data, clim=clim, cmap=cmap, interpolation='none', origin='lower')
    cb = cbar_axes.figure.colorbar(img, cax=cbar_axes, ticks=clim)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    for frame in [on_axes, off_axes]:
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

def plot_rts_summary(rf_data, ax_on, ax_off, cmap=DEFAULT_CMAP):
    rts_on = rf_data['on']['rts']['data']
    rts_off = rf_data['off']['rts']['data']
    plot_fields(rts_on, rts_off, ax_on, ax_off, cmap=cmap)

def plot_rts_blur_summary(rf_data, ax_on, ax_off, cmap=DEFAULT_CMAP):
    rts_on_blur = rf_data['on']['rts_convolution']['data']
    rts_off_blur = rf_data['off']['rts_convolution']['data']
    plot_fields(rts_on_blur, rts_off_blur, ax_on, ax_off, cmap=cmap)

def plot_p_values(rf_data, ax_on, ax_off, cmap=DEFAULT_CMAP):
    pvalues_on = rf_data['on']['pvalues']['data']
    pvalues_off = rf_data['off']['pvalues']['data']
    clim_max = max(pvalues_on.max(), pvalues_off.max())
    plot_fields(pvalues_on, pvalues_off, ax_on, ax_off, clim=(0, clim_max/2), cmap=cmap)

def plot_mask(rf_data, ax_on, ax_off, cmap=DEFAULT_CMAP):
    pvalues_on = rf_data['on']['pvalues']['data']
    pvalues_off = rf_data['off']['pvalues']['data']

    rf_on = pvalues_on.copy()
    rf_off = pvalues_off.copy()

    rf_on[np.logical_not(rf_data['on']['fdr_mask']['data'].sum(axis=0))] = np.nan
    rf_off[np.logical_not(rf_data['off']['fdr_mask']['data'].sum(axis=0))] = np.nan

    plot_fields(rf_on, rf_off, ax_on, ax_off, cmap=cmap)

def plot_gaussian_fit(rf_data, ax_on, ax_off, cmap=DEFAULT_CMAP):

    gf_on_exists = 'gaussian_fit' in rf_data['on']
    gf_off_exists = 'gaussian_fit' in rf_data['off']

    if not gf_on_exists and not gf_off_exists:
        return

    img_data_on = rf_data['on']['gaussian_fit']['data'].sum(axis=0) if gf_on_exists else None
    img_data_off = rf_data['off']['gaussian_fit']['data'].sum(axis=0) if gf_off_exists else None

    if img_data_on and img_data_off:
        plot_fields(img_data_on, img_data_off, ax_on, ax_off, cmap=cmap)
    else:
        if img_data_on:
            img_data_off = np.zeros(img_data_on.shape)
        else:
            img_data_on = np.zeros(img_data_off.shape)

        plot_fields(img_data_on, img_data_off, ax_on, ax_off, cmap=cmap)

def plot_receptive_field_data(receptive_field_data_dict, lsn, show=True, save_file_name=None, close=True, cmap=DEFAULT_CMAP):
    cell_index = receptive_field_data_dict['attrs']['cell_index']

    # Prepare plotting figure:
    number_of_major_rows = 7
    fig = plt.figure(figsize=(4*1.9, number_of_major_rows*2))
    gs = gridspec.GridSpec(number_of_major_rows*2,4)
    ax_list = []

    # Plot chi-square summary:
    curr_axes = fig.add_subplot(gs[0:2, 1:3])
    ax_list += [curr_axes]
    plot_chi_square_summary(receptive_field_data_dict, ax=curr_axes, cmap=cmap)

    # MSR plot:
    if not lsn is None:
        curr_on_axes = fig.add_subplot(gs[2:4, 0:2])
        curr_off_axes = fig.add_subplot(gs[2:4, 2:4])
        ax_list += [curr_on_axes, curr_off_axes]
        plot_msr_summary(lsn, cell_index, curr_on_axes, curr_off_axes, cmap=cmap)

    # RTS no blur:
    curr_on_axes = fig.add_subplot(gs[4:6, 0:2])
    curr_off_axes = fig.add_subplot(gs[4:6, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_rts_summary(receptive_field_data_dict, curr_on_axes, curr_off_axes, cmap=cmap)

    # RTS no blur:
    curr_on_axes = fig.add_subplot(gs[6:8, 0:2])
    curr_off_axes = fig.add_subplot(gs[6:8, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_rts_blur_summary(receptive_field_data_dict, curr_on_axes, curr_off_axes, cmap=cmap)

    # PValues:
    curr_on_axes = fig.add_subplot(gs[8:10, 0:2])
    curr_off_axes = fig.add_subplot(gs[8:10, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_p_values(receptive_field_data_dict, curr_on_axes, curr_off_axes, cmap=cmap)

    # Mask:
    curr_on_axes = fig.add_subplot(gs[10:12, 0:2])
    curr_off_axes = fig.add_subplot(gs[10:12, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_mask(receptive_field_data_dict, curr_on_axes, curr_off_axes, cmap=cmap)

    # Gaussian fit:
    curr_on_axes = fig.add_subplot(gs[12:14, 0:2])
    curr_off_axes = fig.add_subplot(gs[12:14, 2:4])
    ax_list += [curr_on_axes, curr_off_axes]
    plot_gaussian_fit(receptive_field_data_dict, curr_on_axes, curr_off_axes, cmap=cmap)

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
