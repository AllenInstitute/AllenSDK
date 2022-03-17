import scipy.signal as signal
from scipy.stats import zscore
import numpy as np
import pandas as pd
import warnings
from typing import Iterable, Union, Optional


from allensdk.brain_observatory.\
    multi_stimulus_running_speed.multi_stimulus_running_speed import (
        MultiStimulusRunningSpeed
    )

def get_running_df(
    data,
    time: np.ndarray,
    lowpass: bool = True,
    zscore_threshold=10.0
):
    """
    Given the data from the behavior 'pkl' file object and a 1d
    array of timestamps, compute the running speed. Returns a
    dataframe with the raw voltage data as well as the computed speed
    at each timestamp. By default, the running speed is filtered with
    a 10 Hz Butterworth lowpass filter to remove artifacts caused by
    the rotary encoder.

    Parameters
    ----------
    data
        Deserialized 'behavior pkl' file data
    time: np.ndarray (1d)
        Timestamps for running data measurements
    lowpass: bool (default=True)
        Whether to apply a 10Hz low-pass filter to the running speed
        data.
    zscore_threshold: float
        The threshold to use for removing outlier running speeds which might
        be noise and not true signal.

    Returns
    -------
    pd.DataFrame
        Dataframe with an index of timestamps and the following
        columns:
            "speed": computed running speed
            "dx": angular change, computed during data collection
            "v_sig": voltage signal from the encoder
            "v_in": the theoretical maximum voltage that the encoder
                will reach prior to "wrapping". This should
                theoretically be 5V (after crossing 5V goes to 0V, or
                vice versa). In practice the encoder does not always
                reach this value before wrapping, which can cause
                transient spikes in speed at the voltage "wraps".
        The raw data are provided so that the user may compute their
        own speed from source, if desired.

    Notes
    -----
    Though the angular change is available in the raw data
    (key="dx"), this method recomputes the angular change from the
    voltage signal (key="vsig") due to very specific, low-level
    artifacts in the data caused by the encoder. See method
    docstrings for more detailed information. The raw data is
    included in the final output in case the end user wants to apply
    their own corrections and compute running speed from the raw
    source.
    """





    raw_running_speed = data['raw_data']
    v_sig = raw_running_speed['block0_values'][:,0]
    v_in = raw_running_speed['block0_values'][:,1]
    time = raw_running_speed['block0_values'][:,2]
    dx_raw = raw_running_speed['block1_values'][:,0]
    speed = data['raw_data']['block1_values'][:,0]


    return pd.DataFrame({
        'speed': speed,
        'dx': dx_raw,
        'v_sig': v_sig,
        'v_in': v_in,
    }, index=pd.Index(time, name='timestamps'))