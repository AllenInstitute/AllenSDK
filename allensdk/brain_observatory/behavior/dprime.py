import pandas as pd
import numpy as np
from scipy.stats import norm


def get_hit_rate(hit=None, miss=None, aborted=None, sliding_window=100):

    assert len(hit) == len(miss)
    aborted = np.logical_not(np.array(aborted, dtype=np.bool))
    hit = np.array(hit, dtype=np.bool)[aborted]
    miss = np.array(miss, dtype=np.bool)[aborted]
    hit_responses = np.empty_like(hit, dtype=np.float)
    hit_responses.fill(float('nan'))

    hit_responses[hit] = 1
    hit_responses[miss] = 0
    hit_rate = pd.Series(hit_responses).rolling(window=sliding_window, min_periods=0).mean().values
    
    return hit_rate


def get_false_alarm_rate(correct_reject=None, false_alarm=None, aborted=None, sliding_window=100):

    assert len(correct_reject) == len(false_alarm)
    aborted = np.logical_not(np.array(aborted, dtype=np.bool))
    correct_reject = np.array(correct_reject, dtype=np.bool)[aborted]
    false_alarm = np.array(false_alarm, dtype=np.bool)[aborted]
    false_alarm_responses = np.empty_like(correct_reject, dtype=np.float)
    false_alarm_responses.fill(float('nan'))

    false_alarm_responses[false_alarm] = 1
    false_alarm_responses[correct_reject] = 0
    false_alarm_rate = pd.Series(false_alarm_responses).rolling(window=sliding_window, min_periods=0).mean().values

    return false_alarm_rate


def get_dprime_pipeline(hit_rate, fa_rate, limits=(0.01, 0.99)):
    """ calculates the d-prime for a given hit rate and false alarm rate
    https://en.wikipedia.org/wiki/Sensitivity_index
    Parameters
    ----------
    hit_rate : float
        rate of hits in the True class
    fa_rate : float
        rate of false alarms in the False class
    limits : tuple, optional
        limits on extreme values, which distort. default: (0.01,0.99)
    Returns
    -------
    d_prime
    """
    
    assert len(hit_rate) == len(fa_rate)
    assert limits[0] > 0.0, 'limits[0] must be greater than 0.0'
    assert limits[1] < 1.0, 'limits[1] must be less than 1.0'
    Z = norm.ppf

    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(hit_rate, limits[0], limits[1])
    fa_rate = np.clip(fa_rate, limits[0], limits[1])

    try:
        last_hit_nan = np.where(np.isnan(hit_rate))[0].max()
    except ValueError:
        last_hit_nan = 0

    try:
        last_fa_nan = np.where(np.isnan(fa_rate))[0].max()
    except ValueError:
        last_fa_nan = 0

    last_nan = np.max((last_hit_nan, last_fa_nan))

    d_prime = Z(pd.Series(hit_rate)) - Z(pd.Series(fa_rate))

    # fill all values up to the last nan with nan
    d_prime[:last_nan] = np.nan

    if len(d_prime) == 1:
        # if the result is a 1-length vector, return as a scalar
        return d_prime[0]
    else:
        return d_prime
