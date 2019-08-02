import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator



def extract_trial_windows(
    stimulus_table, stimulus_name, time_step, pre_stimulus_time, post_stimulus_time, 
    num_trials=None, stimulus_index=None, 
    name_field='stimulus_name', index_field='stimulus_index', start_field='Start', end_field='End'
):
    ''' Obtains time domains surrounding stimulus sweep onsets
    
    Parameters
    ----------
    stimulus_table : pandas.DataFrame
        Each row is a stimulus sweep. Columns report stimulus name and parameters for that sweep, as well as its start and end times.
    stimulus_name : str
        Obtain sweeps from stimuli with this name (identifies the kind of stimulus presented).
    stimulus_index : int
        Obtain sweeps from stimuli with this index (used to disambiguate presentations of stimuli with the same name).
    time_step : float
        Specifies the step of the resulting temporal domain (seconds).
    pre_stimulus_time : float
        How far before stimulus onset to begin the temporal domain (seconds).
    post_stimulus_time : float
        How far after stimulus onset to end the temporal domain (exclusive, seconds).
    num_trials : int or None, optional
        A window will be computed for this many sweeps. Default is all available.
    name_field : str, optional
        Column from which to extract stimulus name.
    index_field : str, optional
        Column from which to extract stimulus index.
    start_field : str, optional
        Column from which to extract start times.
    end_field : str, optional
        Column from which to extract end times.
        
    Returns
    -------
    trial_windows : list of numpy.ndarray
        For each trial, an array of timestamps surrounding that trial's onset.
    relative_times : numpy.ndarray
        The basic time domain, centered on 0.
    
    '''

    if stimulus_index is None:
        stimulus_index = np.amin(stimulus_table[stimulus_table[name_field] == stimulus_name][index_field].values)

    trials = stimulus_table[
        (stimulus_table[name_field] == stimulus_name)
        & (stimulus_table[index_field] == stimulus_index)
    ]

    if num_trials is not None:
        trials = trials.iloc[:num_trials, :]
    trials = trials.to_dict('record')

    relative_times = np.arange(-pre_stimulus_time, post_stimulus_time, time_step)
    trial_windows = [relative_times + trial[start_field] for trial in trials]
    
    logging.info('calculated relative timestamps: {} ({} timestamps per trial)'.format(relative_times, len(relative_times)))
    logging.info('setup {} trial windows spanning {} to {}'.format(len(trial_windows), trial_windows[0][0], trial_windows[-1][-1]))
    return trial_windows, relative_times



def identify_lfp_channels(surface_channel, unusable_channels, step=2):
    '''Collect a list of channels to be used in CSD calculation
    
    Parameters
    ----------
    surface_channel : int
        The index of the channel lying on the pia surface.
    unusable_channels : list of int
        Indices of channels that cannot be used for CSD calculation. This could occur because they are reference channels or becuase they were identified as damaged.
    step : int, optional
        Defines the size of a zone of channels which are treated as being of equivalent depth. The algorithm attempts to select one channel from each zone.
    
    Returns
    -------
    lfp_channels : np.ndarray
        Array of channels (one at each depth) containing usable LFP data

    '''

    unusable_channels = set(unusable_channels)
    base_channels = set(np.arange(surface_channel))
    filtered_channels = np.sort(list(base_channels - unusable_channels))
    zones = np.floor(filtered_channels / step).astype(int)

    lfp_channels = []
    for ii, (zone, channel) in enumerate(zip(zones, filtered_channels)):
        if ii == 0 or zones[ii] != zones[ii - 1]:
            lfp_channels.append(channel)
            continue
    lfp_channels = np.array(lfp_channels)
    
    logging.info('selected lfp channels: {}'.format(repr(lfp_channels)))
    return lfp_channels


def get_missing_channels(lfp_channels, step=2):
    ''' If the probe has no good channels at a particular depth, this function will pick a new channel at that depth and 
    identify the nearest depths and their channels.

    Parameters
    ----------
    lfp_channels : numpy.ndarray
        Indices of channels whose LFP data can be used directly 
    step : int, optional
        The number of channels per depth
    
    Returns
    -------
    interp_channels : dict
        Keys are channels to be interpolated. Values are dictionaries identifying the depths (zones) of the neighboring 
        usable channels, the usable channel indices themselves, and the depth of the missing channel.

    '''

    lfp_channels = np.array(lfp_channels)
    lfp_zones = np.floor(lfp_channels / step).astype(int)

    zone_diff = np.diff(lfp_zones)
    zone_skips = np.where(zone_diff > 1)[0]

    interp_channels = {}
    for skip_mindex in zone_skips:
        upper_channel = lfp_channels[skip_mindex + 1]
        upper_zone = upper_channel // step
        lower_channel = lfp_channels[skip_mindex]
        lower_zone = lower_channel // step

        new_zones = np.arange(lower_zone + 1, upper_zone)
        for new_zone in new_zones:
            new_channel = new_zone * step

            interp_channels[new_channel] = {
                'base_channels': [lower_channel, upper_channel],
                'base_zones': [lower_zone, upper_zone],
                'zone': new_zone
            }

    logging.info('found missing channel interpolations: {}'.format(repr(interp_channels)))
    return interp_channels



def accumulate_lfp_data(timestamps, lfp_raw, lfp_channels, trial_windows, extractor_factory=None):
    ''' Extracts slices of LFP data at defined channels and times.

    Parameters
    ----------
    timestamps : numpy.ndarray
        Associates LFP sample indices with times in seconds.
    lfp_raw : numpy.ndarray
        Dimensions are samples X channels.
    lfp_channels : numpy.ndarray
        Indices of channels to be used in accumulation
    trial_windows : list of numpy.ndarray
        Each window is a list of times from which LFP data will be extracted.

    Returns
    -------
    accumulated : numpy.ndarray
        Extracted data. Dimensions are trials X channels X samples

    '''

    if extractor_factory is None:
        extractor_factory = regular_grid_extractor_factory
    
    num_samples = min(len(tw) for tw in trial_windows)
    num_trials = len(trial_windows)
    num_channels = len(lfp_channels)

    accumulated = np.zeros((num_trials, num_channels, num_samples), dtype=lfp_raw.dtype)

    for channel_idx, chan in enumerate(lfp_channels):
        logging.info('extracting lfp for channel {}'.format(chan))
        extractor = extractor_factory(timestamps, lfp_raw, chan)

        for trial_index, trial_window in enumerate(trial_windows):
            current = extractor(trial_window)[:num_samples]

            if np.issubdtype(accumulated.dtype, np.integer):
                current = np.around(current).astype(accumulated.dtype)
            accumulated[trial_index, channel_idx, :] = current
    
    logging.info('extracted lfp data for {} trials, {} channels, and {} samples'.format(*accumulated.shape))
    return accumulated
    

def regular_grid_extractor_factory(timestamps, lfp_raw, channel, method='linear'):
    ''' Builds an lfp data extractor using interpolation on a regular grid
    '''
    return RegularGridInterpolator((timestamps,), lfp_raw[:, channel], method=method)

    
def compute_csd(accumulated_lfp, channels=None, missing_channels=None, spacing=0.04):
    ''' Compute current source density for a subset of channels on a neuropixels probe.
        ref: Calculate CSD based on Stoelzel et al. (Swadlow) 2009. 
             Positive is sink and negative is source.
             duplicate first and last depth as per Stoelzel et al. (Swadlow) 2009.

    Parameters
    ----------
    accumulated_lfp : numpy.ndarray
        Stack of extracted LFP traces surrounding presentation of a common stimulus. Dimensions are trials X channels X samples.
        
    spacing: float
        vertical spacing between channels, in milimeter.
        TODO: spacing needs to be inferred through LFP channel id spacing

    Returns
    -------
    csd : numpy.ndarray
        current source density. Dimensions are channels X samples

    '''

    accumulated_lfp = accumulated_lfp - np.mean(accumulated_lfp, axis=2)[:, :, None]
    lfp = np.mean(accumulated_lfp, axis=0)

    lfp, channels = resample_to_regular(lfp, channels, missing_channels)

    lfp_smooth=np.zeros([np.shape(lfp)[0] - 2, np.shape(lfp)[1]])
    csd = np.zeros([np.shape(lfp_smooth)[0] - 2, np.shape(lfp_smooth)[1]])
    
    for t in range(np.shape(lfp)[1]):  # time
        for d in range(1, np.shape(lfp)[0] - 1):  # depth
            lfp_smooth[d - 1, t] = 0.25 * ( lfp[d - 1, t] + 2 * lfp[d, t] + lfp[d + 1, t] )
    
    for t in range(np.shape(lfp_smooth)[1]): # time
        for d in np.arange(1, np.shape(lfp_smooth)[0] - 1):
            csd[d - 1, t]=(1 / spacing ** 2) * ( lfp_smooth[d + 1, t] - 2 * lfp_smooth[d, t] + lfp_smooth[d - 1, t] )    
    
    return csd, channels


def resample_to_regular(lfp, channels=None, missing_channels=None):
    ''' Linearly interpolates in order to fill in channel gaps in a 2D LFP array.
    
    Parameters
    ----------
    lfp : numpy.ndarray
        2D, with dimensions channel X sample
    channels : numpy.ndarray, optional
        Indices of channels whose LFP data can be used directly 
    missing_channels : dict, optional
        get_missing_channels output
    
    Returns
    -------
    new_lfp : numpy.ndarray
        Same dimensions as input LFP, but missing channel data has been interpolated.
    total_channels : numpy.ndarray
        Array of channel indices, real and interpolated, that correspond to rows of the interpolated LFP image.

    '''

    if channels is None:
        return lfp, np.arange(lfp.shape[0])
    if missing_channels is None:
        return lfp, channels

    channel_indices = {ch: ii for ii, ch in enumerate(channels)}
    total_channels = sorted(list(channels) + list(missing_channels.keys()))
    new_lfp = np.zeros([len(total_channels), lfp.shape[1]])

    for ii, channel in enumerate(total_channels):
        if channel in channel_indices:
            old_index = channel_indices[channel]
            new_lfp[ii, :] = lfp[old_index, :]
        else:
            lower_zone, upper_zone = missing_channels[channel]['base_zones']
            lower_channel, upper_channel = missing_channels[channel]['base_channels']
            lower_index = channel_indices[lower_channel]
            upper_index = channel_indices[upper_channel]

            logging.info('interpolating missing channel {} from channels {} and {}'.format(
                channel, lower_channel, upper_channel
            ))

            weight = (missing_channels[channel]['zone'] - lower_zone) / (upper_zone - lower_zone)
            new_lfp[ii, :] = weight * lfp[lower_index, :] + (1 - weight) * lfp[upper_index, :]

    logging.info('interpolated lfp to {} channels and {} samples'.format(*new_lfp.shape))
    logging.info('total channels: {}'.format(repr(total_channels)))
    return new_lfp, np.array(total_channels)
