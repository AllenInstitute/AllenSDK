from scipy.ndimage import percentile_filter
from tqdm import tqdm
from multiprocessing import Pool

def nanmedian_filter(x,filter_length):
    """ 1D median filtering with np.nanmedian
    Parameters
    ----------
    x: 1D trace to be filtered
    filter_length: length of the filter

    Return
    ------
    filtered_trace
    """
    half_length = int(filter_length/2)
    # Create 'reflect' traces at the extrema
    temp_trace = np.concatenate((np.flip(x[:half_length]), x, np.flip(x[-half_length:])))
    filtered_trace = np.zeros_like(x)
    for i in range(len(x)):
        filtered_trace[i] = np.nanmedian(temp_trace[i:i+filter_length])
    return filtered_trace

def noise_std(x, filter_length, positive_peak_scale=1.5,
    outlier_std_scale=2.5, GAUSSIAN_MAD_STD_SCALE = 1.4826):
    """Robust estimate of the standard deviation of the trace noise."""
    if any(np.isnan(x)):
        return np.NaN
    x = x - nanmedian_filter(x, filter_length)
    # first pass removing big pos peak outliers
    x = x[x < positive_peak_scale*np.abs(x.min())]

    rstd = GAUSSIAN_MAD_STD_SCALE*np.median(np.abs(x - np.median(x)))
    # second pass removing remaining pos and neg peak outliers
    x = x[abs(x) < outlier_std_scale*rstd]
    x = GAUSSIAN_MAD_STD_SCALE*np.median(np.abs(x - np.median(x)))
    return x

def compute_dff_single_trace(roi_ind,corrected_trace, 
                            frames_per_sec, inactive_kernel_size = 30,  
                            inactive_percentile = 10):
    
    long_filter_length = int(round(frames_per_sec*60*inactive_kernel_size))
    short_filter_length = int(round(frames_per_sec*60*10)) # 10 min
    

    noise_sd = noise_std(corrected_trace, filter_length=int(round(frames_per_sec * 3.33))) # 3.33 s is fixed
    # 10th percentile "low_baseline"
    low_baseline = percentile_filter(corrected_trace, size=long_filter_length, percentile=inactive_percentile, mode='reflect')
    # Create trace using inactive frames only, by replacing signals in "active frames" with NaN
    active_frame = np.where(corrected_trace > (low_baseline + 3 * noise_sd))[0] # type: ignore
    inactive_trace = corrected_trace.copy()
    for i in active_frame:
        inactive_trace[i] = np.nan
    # Calculate baseline using median filter
    baseline_new = nanmedian_filter(inactive_trace, short_filter_length)
    # Calculate DFF

    dff_trace = ((corrected_trace - baseline_new)/ np.maximum(baseline_new, noise_sd))
    dff_trace[np.argwhere(np.isnan(dff_trace))] = 0
    return dff_trace

def compute_dff(corrected_fluorescence_traces, ophys_timestamps):

    frame_per_sec = 1/np.nanmean((np.diff(ophys_timestamps)))
    dff_traces = corrected_fluorescence_traces.copy()
    corrected = np.array(dff_traces['corrected_fluorescence'])
    dff = corrected.copy()
    
    cores = 15 # cpu cores to be used, optimized so it won't max out the memory

    for i_core in tqdm(np.arange(0,np.shape(corrected)[0],cores), desc=''):
        args = [(roi_ind, corrected[roi_ind], frame_per_sec) for roi_ind in np.arange(i_core,np.min((i_core+cores,np.shape(corrected)[0])))]
    
        with Pool() as pool:
            tmp = pool.starmap(compute_dff_single_trace, args)
        
        dff[i_core:np.min((i_core+cores,np.shape(corrected)[0]))] = tmp
    dff_traces.pop('RMSE')
    dff_traces.pop('r')
    dff_traces.pop('corrected_fluorescence')
    dff_traces['dff'] = dff

    return(dff_traces)