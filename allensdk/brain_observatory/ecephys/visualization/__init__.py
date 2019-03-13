from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np


def plot_mean_waveforms(mean_waveforms, unit_ids): # pragma: no cover
    ''' Utility for plotting mean waveforms on each unit's peak channel

    Parameters
    ----------
    mean_waveforms : dictionary
        Maps unit ids to channelwise averege spike waveforms for those units
    unit_ids : array-like
        unique integer identifiers for units to be included

    '''

    fig, ax = plt.subplots(figsize=(10, 10))

    for uid in unit_ids:
        wf = mean_waveforms[uid]
        delta = wf.max(dim='time') - wf.min(dim='time') # TODO: there is something wrong with the peak channels in the nwb file, so we are computing them here as a stopgap
        ch_max = delta.where(delta == delta.max(), drop=True)['channel_id'][0]
        ax.plot(wf.loc[{'channel_id': ch_max}])

    ax.legend(unit_ids)
    ax.set_ylabel('membrane potential (mV)', fontsize=16)
    ax.set_xlabel('time (s)', fontsize=16)

    ax.set_xticks(np.arange(0, len(wf['time']), 20))
    ax.set_xticklabels([f'{float(ii):1.4f}' for ii in wf['time'][::20]], rotation=45)

    return fig
    

def plot_spike_counts(
    data_array, 
    time_coords,
    cbar_label, 
    title, 
    xlabel='time relative to stimulus onset (s)', 
    ylabel='unit', 
    xtick_step=20
): # pragma: no cover
    '''Utility for making a simple spike counts plot.

    Parameters
    ----------
    data_array : xarray.DataArray
        2D data array unitwise values per time bin. See EcephysSession.sweepwise_spike_counts

    '''
    
    fig, ax = plt.subplots(figsize=(12, 12))
    div = make_axes_locatable(ax)
    cbar_axis = div.append_axes("right", 0.2, pad=0.05)

    img = ax.imshow(
        data_array.T, 
        interpolation='none'
    )
    plt.colorbar(img, cax=cbar_axis)

    cbar_axis.set_ylabel(cbar_label, fontsize=16)

    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel(ylabel, fontsize=16)

    reltime = np.array(time_coords)
    ax.set_xticks(np.arange(0, len(reltime), xtick_step))
    ax.set_xticklabels([f'{mp:1.3f}' for mp in reltime[::xtick_step]], rotation=45)
    ax.set_xlabel(xlabel, fontsize=16)

    ax.set_title(title, fontsize=20)

    return fig