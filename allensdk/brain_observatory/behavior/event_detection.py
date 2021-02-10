import numpy as np
from scipy import stats


def filter_events_array(arr: np.ndarray, scale: float = 2, t_scale: int = 20) -> np.ndarray:
    """
    Convolve the trace array with a 1d causal half-gaussian filter
    to smooth it for visualization

    Uses a halfnorm distribution as weights to the filter

    Author: Nick Ponvert

    Parameters
    ----------
    arr: np.ndarray
        Trace matrix of dimension n traces x n frames
    scale: float
        std deviation of halfnorm distribution
    t_scale: int
        time scale to use for the convolution operation

    Returns
    ----------
    np.ndarray:
        Output of the convolution operation
    """
    filt = stats.halfnorm(loc=0, scale=scale).pdf(np.arange(t_scale))
    filt = filt / np.sum(filt)  # normalize filter
    filtered_arr = np.zeros(arr.shape)
    for i, trace in enumerate(arr):
        filtered_arr[i] = np.convolve(arr[i], filt)[:len(arr[i])]
    return filtered_arr
