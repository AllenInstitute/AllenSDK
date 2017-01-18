# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import warnings
import logging
import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit
from functools import partial

def detect_putative_spikes(v, t, start=None, end=None, filter=10., dv_cutoff=20.):
    """Perform initial detection of spikes and return their indexes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of time window for spike detection (optional)
    end : end of time window for spike detection (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dv_cutoff : minimum dV/dt to qualify as a spike in V/s (optional, default 20)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    putative_spikes : numpy array of preliminary spike indexes
    """

    if not isinstance(v, np.ndarray):
        raise TypeError("v is not an np.ndarray")

    if not isinstance(t, np.ndarray):
        raise TypeError("t is not an np.ndarray")

    if v.shape != t.shape:
        raise FeatureError("Voltage and time series do not have the same dimensions")

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    start_index = find_time_index(t, start)
    end_index = find_time_index(t, end)
    v_window = v[start_index:end_index + 1]
    t_window = t[start_index:end_index + 1]

    dvdt = calculate_dvdt(v_window, t_window, filter)

    # Find positive-going crossings of dV/dt cutoff level
    putative_spikes = np.flatnonzero(np.diff(np.greater_equal(dvdt, dv_cutoff).astype(int)) == 1)

    if len(putative_spikes) <= 1:
        # Set back to original index space (not just window)
        return np.array(putative_spikes) + start_index

    # Only keep spike times if dV/dt has dropped all the way to zero between putative spikes
    putative_spikes = [putative_spikes[0]] + [s for i, s in enumerate(putative_spikes[1:])
        if np.any(dvdt[putative_spikes[i]:s] < 0)]

    # Set back to original index space (not just window)
    return np.array(putative_spikes) + start_index


def find_peak_indexes(v, t, spike_indexes, end=None):
    """Find indexes of spike peaks.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    end : end of time window for spike detection (optional)
    """

    if not end:
        end = t[-1]
    end_index = find_time_index(t, end)

    spks_and_end = np.append(spike_indexes, end_index)
    peak_indexes = [np.argmax(v[spk:next]) + spk for spk, next in
                    zip(spks_and_end[:-1], spks_and_end[1:])]

    return np.array(peak_indexes)


def filter_putative_spikes(v, t, spike_indexes, peak_indexes, min_height=2., min_peak=-30.):
    """Filter out events that are unlikely to be spikes based on:
        * Height (threshold to peak)
        * Absolute peak level

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    min_height : minimum acceptable height from threshold to peak in mV (optional, default 2)
    min_peak : minimum acceptable absolute peak level in mV (optional, default -30)

    Returns
    -------
    spike_indexes : numpy array of threshold indexes
    peak_indexes : numpy array of peak indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([]), np.array([])

    peak_level_mask = v[peak_indexes] >= min_peak
    spike_indexes = spike_indexes[peak_level_mask]
    peak_indexes = peak_indexes[peak_level_mask]

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= min_height
    spike_indexes = spike_indexes[height_mask]
    peak_indexes = peak_indexes[height_mask]

    return spike_indexes, peak_indexes


def find_upstroke_indexes(v, t, spike_indexes, peak_indexes, filter=10., dvdt=None):
    """Find indexes of maximum upstroke of spike.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    upstroke_indexes : numpy array of upstroke indexes

    """

    if dvdt is None:
        dvdt = calculate_dvdt(v, t, filter)

    upstroke_indexes = [np.argmax(dvdt[spike:peak]) + spike for spike, peak in
                        zip(spike_indexes, peak_indexes)]

    return np.array(upstroke_indexes)


def refine_threshold_indexes(v, t, upstroke_indexes, thresh_frac=0.05, filter=10., dvdt=None):
    """Refine threshold detection of previously-found spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    upstroke_indexes : numpy array of indexes of spike upstrokes (for threshold target calculation)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    threshold_indexes : numpy array of threshold indexes
    """

    if not upstroke_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = calculate_dvdt(v, t, filter)

    avg_upstroke = dvdt[upstroke_indexes].mean()
    target = avg_upstroke * thresh_frac

    upstrokes_and_start = np.append(np.array([0]), upstroke_indexes)
    threshold_indexes = []
    for upstk, upstk_prev in zip(upstrokes_and_start[1:], upstrokes_and_start[:-1]):
        potential_indexes = np.flatnonzero(dvdt[upstk:upstk_prev:-1] <= target)
        if not potential_indexes.size:
            # couldn't find a matching value for threshold,
            # so just going to the start of the search interval
            threshold_indexes.append(upstk_prev)
        else:
            threshold_indexes.append(upstk - potential_indexes[0])

    return np.array(threshold_indexes)


def check_thresholds_and_peaks(v, t, spike_indexes, peak_indexes, upstroke_indexes,
                               max_interval=0.005, thresh_frac=0.05, filter=10., dvdt=None):
    """Validate thresholds and peaks for set of spikes

    Check that peaks and thresholds for consecutive spikes do not overlap
    Spikes with overlapping thresholds and peaks will be merged.

    Check that peaks and thresholds for a given spike are not too far apart.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    upstroke_indexes : numpy array of indexes of spike upstrokes
    max_interval : maximum allowed time between start of spike and time of peak in sec (default 0.005)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    spike_indexes : numpy array of modified spike indexes
    peak_indexes : numpy array of modified spike peak indexes
    upstroke_indexes : numpy array of modified spike upstroke indexes
    """

    overlaps = np.flatnonzero(spike_indexes[1:] <= peak_indexes[:-1])
    if overlaps.size:
        spike_mask = np.ones_like(spike_indexes, dtype=bool)
        spike_mask[overlaps + 1] = False
        spike_indexes = spike_indexes[spike_mask]

        peak_mask = np.ones_like(peak_indexes, dtype=bool)
        peak_mask[overlaps] = False
        peak_indexes = peak_indexes[peak_mask]

        upstroke_mask = np.ones_like(upstroke_indexes, dtype=bool)
        upstroke_mask[overlaps] = False
        upstroke_indexes = upstroke_indexes[upstroke_mask]

    # Validate that peaks don't occur too long after the threshold
    # If they do, try to re-find threshold from the peak
    too_long_spikes = []
    for i, (spk, peak) in enumerate(zip(spike_indexes, peak_indexes)):
        if t[peak] - t[spk] >= max_interval:
            logging.info("Need to recalculate threshold-peak pair that exceeds maximum allowed interval ({:f} s)".format(max_interval))
            too_long_spikes.append(i)

    if too_long_spikes:
        if dvdt is None:
            dvdt = calculate_dvdt(v, t, filter)
        avg_upstroke = dvdt[upstroke_indexes].mean()
        target = avg_upstroke * thresh_frac
        drop_spikes = []
        for i in too_long_spikes:
            # First guessing that threshold is wrong and peak is right
            peak = peak_indexes[i]
            t_0 = find_time_index(t, t[peak] - max_interval)
            below_target = np.flatnonzero(dvdt[upstroke_indexes[i]:t_0:-1] <= target)
            if not below_target.size:
                # Now try to see if threshold was right but peak was wrong

                # Find the peak in a window twice the size of our allowed window
                spike = spike_indexes[i]
                t_0 = find_time_index(t, t[spike] + 2 * max_interval)
                new_peak = np.argmax(v[spike:t_0]) + spike

                # If that peak is okay (not outside the allowed window, not past the next spike)
                # then keep it
                if t[new_peak] - t[spike] < max_interval and \
                   (i == len(spike_indexes) - 1 or t[new_peak] < t[spike_indexes[i + 1]]):
                    peak_indexes[i] = new_peak
                else:
                    # Otherwise, log and get rid of the spike
                    logging.info("Could not redetermine threshold-peak pair - dropping that pair")
                    drop_spikes.append(i)
#                     raise FeatureError("Could not redetermine threshold")
            else:
                spike_indexes[i] = upstroke_indexes[i] - below_target[0]

        if drop_spikes:
            spike_indexes = np.delete(spike_indexes, drop_spikes)
            peak_indexes = np.delete(peak_indexes, drop_spikes)
            upstroke_indexes = np.delete(upstroke_indexes, drop_spikes)

    return spike_indexes, peak_indexes, upstroke_indexes


def find_trough_indexes(v, t, spike_indexes, peak_indexes, end=None):
    """
    Find indexes of minimum voltage (trough) between spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)

    Returns
    -------
    trough_indexes : numpy array of threshold indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if end is None:
        end = t[-1]
    end_index = find_time_index(t, end)

    trough_indexes = np.zeros_like(spike_indexes)
    trough_indexes[:-1] = [v[peak:spk].argmin() + peak for peak, spk in zip(peak_indexes[:-1], spike_indexes[1:])]

    if peak_indexes[-1] >= end_index:
        # If peak of last spike is the last point available, drop last point (since no trough exists)
        trough_indexes = trough_indexes[:-1]
    else:
        trough_indexes[-1] = v[peak_indexes[-1]:end_index].argmin() + peak_indexes[-1]

    # If peak is the same point as the trough, drop that point
    trough_indexes = trough_indexes[np.where(peak_indexes[:len(trough_indexes)] != trough_indexes)]

    return trough_indexes


def find_downstroke_indexes(v, t, peak_indexes, trough_indexes, filter=10., dvdt=None):
    """Find indexes of minimum voltage (troughs) between spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of threshold indexes
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    downstroke_indexes : numpy array of downstroke indexes
    """

    if not trough_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = calculate_dvdt(v, t, filter)

    if len(peak_indexes) < len(trough_indexes):
        raise FeatureError("Cannot have more troughs than peaks")

    peak_indexes = peak_indexes[:len(trough_indexes)]

    # Can't handle cases where peak is
    downstroke_indexes = [np.argmin(dvdt[peak:trough]) + peak for peak, trough
                         in zip(peak_indexes, trough_indexes)]

    return np.array(downstroke_indexes)


def find_widths(v, t, spike_indexes, peak_indexes, trough_indexes):
    """Find widths at half-height for spikes.

    Widths are only returned when heights are defined

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of trough indexes

    Returns
    -------
    widths : numpy array of spike widths in sec
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if len(spike_indexes) < len(trough_indexes):
        raise FeatureError("Cannot have more troughs than spikes")

    use_indexes = ~np.isnan(trough_indexes)

    heights = np.zeros_like(trough_indexes) * np.nan
    heights[use_indexes] = v[peak_indexes[use_indexes]] - v[trough_indexes[use_indexes].astype(int)]

    width_levels = np.zeros_like(trough_indexes) * np.nan
    width_levels[use_indexes] = heights[use_indexes] / 2. + v[trough_indexes[use_indexes].astype(int)]

    thresh_to_peak_levels = np.zeros_like(trough_indexes) * np.nan
    thresh_to_peak_levels[use_indexes] = (v[peak_indexes[use_indexes]] - v[spike_indexes[use_indexes]]) / 2. + v[spike_indexes[use_indexes]]

    # Some spikes in burst may have deep trough but short height, so can't use same
    # definition for width
    width_levels[width_levels < v[spike_indexes]] = \
        thresh_to_peak_levels[width_levels < v[spike_indexes]]

    width_starts = np.zeros_like(trough_indexes) * np.nan
    width_starts[use_indexes] = np.array([pk - np.flatnonzero(v[pk:spk:-1] <= wl)[0] if
                    np.flatnonzero(v[pk:spk:-1] <= wl).size > 0 else np.nan for pk, spk, wl
                    in zip(peak_indexes[use_indexes], spike_indexes[use_indexes], width_levels[use_indexes])])
    width_ends = np.zeros_like(trough_indexes) * np.nan
    width_ends[use_indexes] = np.array([pk + np.flatnonzero(v[pk:tr] <= wl)[0] if
                    np.flatnonzero(v[pk:tr] <= wl).size > 0 else np.nan for pk, tr, wl
                    in zip(peak_indexes[use_indexes], trough_indexes[use_indexes], width_levels[use_indexes])])

    missing_widths = np.isnan(width_starts) | np.isnan(width_ends)
    widths = np.zeros_like(width_starts, dtype=np.float64)
    widths[~missing_widths] = t[width_ends[~missing_widths].astype(int)] - \
                              t[width_starts[~missing_widths].astype(int)]
    if any(missing_widths):
        widths[missing_widths] = np.nan

    return widths


def analyze_trough_details(v, t, spike_indexes, peak_indexes, end=None, filter=10.,
                           heavy_filter=1., term_frac=0.01, adp_thresh=0.5, tol=0.5,
                           flat_interval=0.002, dvdt=None):
    """Analyze trough to determine if an ADP exists and whether the reset is a 'detour' or 'direct'

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    heavy_filter : lower cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    adp_thresh: minimum dV/dt in V/s to exceed to be considered to have an ADP (optional, default 1.5)
    tol : tolerance for evaluating whether Vm drops appreciably further after end of spike (default 1.0 mV)
    flat_interval: if the trace is flat for this duration, stop looking for an ADP (default 0.002)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    isi_types : numpy array of isi reset types (direct or detour)
    fast_trough_indexes : numpy array of indexes at the start of the trough (i.e. end of the spike)
    adp_indexes : numpy array of adp indexes (np.nan if there was no ADP in that ISI
    slow_trough_indexes : numpy array of indexes at the minimum of the slow phase of the trough
                          (if there wasn't just a fast phase)
    """

    if end is None:
        end = t[-1]
    end_index = find_time_index(t, end)

    # Can't evaluate peaks that occur at or past end of the analysis interval
    orig_len = len(peak_indexes)
    peak_mask = peak_indexes < end_index
    spike_indexes = spike_indexes[peak_mask]
    peak_indexes = peak_indexes[peak_mask]

    if dvdt is None:
        dvdt = calculate_dvdt(v, t, filter)

    dvdt_hvy = calculate_dvdt(v, t, heavy_filter)

    # Writing as for loop - see if I can vectorize any later
    fast_trough_indexes = []
    adp_indexes = []
    slow_trough_indexes = []
    isi_types = []

    for peak, next_spk in zip(peak_indexes, np.append(spike_indexes[1:], end_index)):
        downstroke = dvdt[peak:next_spk].argmin() + peak
        target = term_frac * dvdt[downstroke]

        terminated_points = np.flatnonzero(dvdt[downstroke:next_spk] >= target)
        if terminated_points.size:
            terminated = terminated_points[0] + downstroke
        else:
            isi_types.append(np.nan)
            fast_trough_indexes.append(np.nan)
            adp_indexes.append(np.nan)
            slow_trough_indexes.append(np.nan)
            continue

        # Could there be an ADP?
        adp_index = np.nan
        dv_over_thresh = np.flatnonzero(dvdt_hvy[terminated:next_spk] >= adp_thresh)
        if dv_over_thresh.size:
            cross = dv_over_thresh[0] + terminated

            # only want to look for ADP before things get pretty flat
            # otherwise, could just pick up random transients long after the spike
            if t[cross] - t[terminated] < flat_interval:
                # Going back up fast, but could just be going into another spike
                # so need to check for a reversal (zero-crossing) in dV/dt
                zero_return_vals = np.flatnonzero(dvdt_hvy[cross:next_spk] <= 0)
                if zero_return_vals.size:
                    putative_adp_index = zero_return_vals[0] + cross
                    min_index = v[putative_adp_index:next_spk].argmin() + putative_adp_index
                    if v[putative_adp_index] - v[min_index] >= tol:
                        adp_index = putative_adp_index
                        slow_phase_min_index = min_index
                        isi_type = "detour"

        if np.isnan(adp_index):
            v_term = v[terminated]
            min_index = v[terminated:next_spk].argmin() + terminated
            if v_term - v[min_index] >= tol:
                # dropped further after end of spike -> detour reset
                isi_type = "detour"
                slow_phase_min_index = min_index
            else:
                isi_type = "direct"

        isi_types.append(isi_type)
        fast_trough_indexes.append(terminated)
        adp_indexes.append(adp_index)
        if isi_type == "detour":
            slow_trough_indexes.append(slow_phase_min_index)
        else:
            slow_trough_indexes.append(np.nan)

    # If we had to kick some spikes out before, need to add nans at the end
    output = map(np.array, (isi_types, fast_trough_indexes, adp_indexes, slow_trough_indexes))
    if orig_len > len(isi_types):
        extra = np.zeros(orig_len - len(isi_types)) * np.nan
        output = tuple((np.append(o, extra) for o in output))
    return output


def find_time_index(t, t_0):
    """Find the index value of a given time (t_0) in a time series (t)."""

    t_gte = np.flatnonzero(t >= t_0)
    if not t_gte.size:
        raise FeatureError("Could not find given time in time vector")

    return t_gte[0]


def calculate_dvdt(v, t, filter=None):
    """Low-pass filters (if requested) and differentiates voltage by time.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default None)

    Returns
    -------
    dvdt : numpy array of time-derivative of voltage (V/s = mV/ms)
    """

    if has_fixed_dt(t) and filter:
        delta_t = t[1] - t[0]
        sample_freq = 1. / delta_t
        filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency
        if filt_coeff < 0 or filt_coeff > 1:
            raise FeatureError("bessel coeff (%f) is outside of valid range [0,1]. cannot compute features." % filt_coeff)
        b, a = signal.bessel(4, filt_coeff, "low")
        v_filt = signal.filtfilt(b, a, v, axis=0)
        dv = np.diff(v_filt)
    else:
        dv = np.diff(v)

    dt = np.diff(t)
    dvdt = 1e-3 * dv / dt # in V/s = mV/ms

    # Remove nan values (in case any dt values == 0)
    dvdt = dvdt[~np.isnan(dvdt)]

    return dvdt


def get_isis(t, spikes):
    """Find interspike intervals in sec between spikes (as indexes)."""

    if len(spikes) <= 1:
        return np.array([])

    return t[spikes[1:]] - t[spikes[:-1]]


def average_voltage(v, t, start=None, end=None):
    """Calculate average voltage between start and end.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of time window for spike detection (optional, default None)
    end : end of time window for spike detection (optional, default None)

    Returns
    -------
    v_avg : average voltage
    """

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    start_index = find_time_index(t, start)
    end_index = find_time_index(t, end)

    return v[start_index:end_index].mean()


def adaptation_index(isis):
    """Calculate adaptation index of `isis`."""
    if len(isis) == 0:
        return np.nan

    return norm_diff(isis)


def latency(t, spikes, start):
    """Calculate time to the first spike."""

    if len(spikes) == 0:
        return np.nan

    if start is None:
        start = t[0]

    return t[spikes[0]] - start


def average_rate(t, spikes, start, end):
    """Calculate average firing rate during interval between `start` and `end`.

    Parameters
    ----------
    t : numpy array of times in seconds
    spikes : numpy array of spike indexes
    start : start of time window for spike detection
    end : end of time window for spike detection

    Returns
    -------
    avg_rate : average firing rate in spikes/sec
    """

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    spikes_in_interval = [spk for spk in spikes if t[spk] >= start and t[spk] <= end]
    avg_rate = len(spikes_in_interval) / (end - start)
    return avg_rate


def norm_diff(a):
    """Calculate average of (a[i] - a[i+1]) / (a[i] + a[i+1])."""

    if len(a) <= 1:
        return np.nan

    a = a.astype(float)
    if np.allclose((a[1:] + a[:-1]), 0.):
        return 0.
    norm_diffs = (a[1:] - a[:-1]) / (a[1:] + a[:-1])
    norm_diffs[(a[1:] == 0) & (a[:-1] == 0)] = 0.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        avg = np.nanmean(norm_diffs)
    return avg


def norm_sq_diff(a):
    """Calculate average of (a[i] - a[i+1])^2 / (a[i] + a[i+1])^2."""
    if len(a) <= 1:
        return np.nan

    a = a.astype(float)
    norm_sq_diffs = np.square((a[1:] - a[:-1])) / np.square((a[1:] + a[:-1]))
    return norm_sq_diffs.mean()


def has_fixed_dt(t):
    """Check that all time intervals are identical."""
    dt = np.diff(t)
    return np.allclose(dt, np.ones_like(dt) * dt[0])


def fit_membrane_time_constant(v, t, start, end):
    """Fit an exponential to estimate membrane time constant between start and end

    Parameters
    ----------
    v : numpy array of voltages in mV
    t : numpy array of times in seconds
    start : start of time window for exponential fit
    end : end of time window for exponential fit

    Returns
    -------
    a, inv_tau, y0 : Coeffients of equation y0 + a * exp(-inv_tau * x)

    returns np.nan for values if fit fails
    """

    start_index = find_time_index(t, start)
    end_index = find_time_index(t, end)

    guess = (v[start_index] - v[end_index], 50., v[end_index])
    t_window = (t[start_index:end_index] - t[start_index]).astype(np.float64)
    v_window = v[start_index:end_index].astype(np.float64)
    try:
        popt, pcov = curve_fit(_exp_curve, t_window, v_window, p0=guess)
    except RuntimeError:
        logging.info("Curve fit for membrane time constant failed")
        return np.nan, np.nan, np.nan

    return popt


def detect_pauses(isis, isi_types, cost_weight=1.0):
    """Determine which ISIs are "pauses" in ongoing firing.

    Pauses are unusually long ISIs with a "detour reset" among "direct resets".

    Parameters
    ----------
    isis : numpy array of interspike intervals
    isi_types : numpy array of interspike interval types ('direct' or 'detour')
    cost_weight : weight for cost function for calling an ISI a pause
        Higher cost weights lead to fewer ISIs identified as pauses. The cost function
        also depends on the difference between the duration of the "pause" ISIs and the
        average duration and standard deviation of "non-pause" ISIs.

    Returns
    -------
    pauses : numpy array of indices corresponding to pauses in `isis`
    """

    if len(isis) != len(isi_types):
        raise FeatureError("Wrong number of ISIs")

    if not np.any(isi_types == "direct"):
        # Need some direct-type firing to have pauses
        return np.array([])

    detour_candidates = [i for i, isi_type in enumerate(isi_types) if isi_type == "detour"]
    median_direct = np.median(isis[isi_types == "direct"])
    direct_candidates = [i for i, isi_type in enumerate(isi_types) if isi_type == "direct" and isis[i] > 3 * median_direct]
    candidates = detour_candidates + direct_candidates

    if not candidates:
        return np.array([])

    pause_list = np.array([], dtype=int)
    all_cv = isis.std() / isis.mean()
    best_net = 0
    for i in candidates:
        temp_pause_list = np.append(pause_list, i)
        non_pause_isis = np.delete(isis, temp_pause_list)
        pause_isis = isis[temp_pause_list]
        if len(non_pause_isis) < 2:
            break
        cv = non_pause_isis.std() / non_pause_isis.mean()
        benefit = all_cv - cv
        cost = np.sum(non_pause_isis.std() / np.abs(non_pause_isis.mean() - pause_isis))
        cost *= cost_weight
        net = benefit - cost
        if net > 0 and net < best_net:
            break
        if net > best_net:
            best_net = net
        pause_list = np.append(pause_list, i)

    if best_net <= 0:
        pause_list = np.array([])

    return np.sort(pause_list)


def detect_bursts(isis, isi_types, fast_tr_v, fast_tr_t, slow_tr_v, slow_tr_t,
                  thr_v, tol=0.5, pause_cost=1.0):
    """Detect bursts in spike train.

    Parameters
    ----------
    isis : numpy array of n interspike intervals
    isi_types : numpy array of n interspike interval types
    fast_tr_v : numpy array of fast trough voltages for the n + 1 spikes of the train
    fast_tr_t : numpy array of fast trough times for the n + 1 spikes of the train
    slow_tr_v : numpy array of slow trough voltages for the n + 1 spikes of the train
    slow_tr_t : numpy array of slow trough times for the n + 1 spikes of the train
    thr_v : numpy array of threshold voltages for the n + 1 spikes of the train
    tol : tolerance for the difference in slow trough voltages and thresholds (default 0.5 mV)
        Used to identify "delay" interspike intervals that occur within a burst

    Returns
    -------
    bursts : list of bursts
        Each item in list is a tuple of the form (burst_index, start, end) where `burst_index`
        is a comparison index between the highest instantaneous rate within the burst vs
        the highest instantaneous rate outside the burst. `start` is the index of the first
        ISI of the burst, and `end` is the ISI index immediately following the burst.
    """

    if len(isis) != len(isi_types):
        raise FeatureError("Wrong number of ISIs")

    if len(isis) < 2: # can't determine burstiness for a single ISI
        return np.array([])

    fast_tr_v = fast_tr_v[:-1]
    fast_tr_t = fast_tr_t[:-1]
    slow_tr_v = slow_tr_v[:-1]
    slow_tr_t = slow_tr_t[:-1]

    isi_types = np.array(isi_types) # don't want to change the actual isi types data

    # Burst transitions can't be at "pause"-like ISIs
    pauses = detect_pauses(isis, isi_types, cost_weight=pause_cost).astype(int)
    isi_types[pauses] = "pauselike"

    if not (np.any(isi_types == "direct") and np.any(isi_types == "detour")):
        # no candidates that could be bursts
        return np.array([])

    # Want to catch special case of detour in the middle of a large burst where
    # the slow trough value is higher than the previous spike's threshold
    isi_types[(thr_v[:-1] < (slow_tr_v + tol)) & (isi_types == "detour")] = "midburst"

    # Find transitions from direct -> detour and vice versa for burst boundaries
    into_burst = np.array([i + 1 for i, (prev, cur) in
                 enumerate(zip(isi_types[:-1], isi_types[1:])) if
                 cur == "direct" and prev == "detour"],
                 dtype=int)
    if isi_types[0] == "direct":
        into_burst = np.append(np.array([0]), into_burst)

    drop_into = []
    out_of_burst = []
    for j, (into, next) in enumerate(zip(into_burst, np.append(into_burst[1:], len(isis)))):
        for i, isi in enumerate(isi_types[into + 1:next]):
            if isi == "detour":
                out_of_burst.append(i + into + 1)
                break
            elif isi == "pauselike":
                drop_into.append(j)
                break
    mask = np.ones_like(into_burst, dtype=bool)
    mask[drop_into] = False
    into_burst = into_burst[mask]

    out_of_burst = np.array(out_of_burst)
    if len(out_of_burst) == len(into_burst) - 1:
        out_of_burst = np.append(out_of_burst, len(isi_types))

    if not (into_burst.size or out_of_burst.size):
        return np.array([])

    if len(into_burst) != len(out_of_burst):
        raise FeatureError("Inconsistent burst boundary identification")

    inout_pairs = zip(into_burst, out_of_burst)
    delta_t = slow_tr_t - fast_tr_t

    scores = _score_burst_set(inout_pairs, isis, delta_t)
    best_score = np.mean(scores)
    worst = np.argmin(scores)
    test_bursts = list(inout_pairs)
    del test_bursts[worst]
    while len(test_bursts) > 0:
        scores = _score_burst_set(test_bursts, isis, delta_t)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            inout_pairs = list(test_bursts)
            worst = np.argmin(scores)
            del test_bursts[worst]
        else:
            break

    if best_score < 0:
        return np.array([])

    bursts = []
    for i, (into, outof) in enumerate(inout_pairs):
        if i == len(inout_pairs) - 1: # last burst to evaluate
            if outof <= len(isis) - 1: # are there spikes left after the burst?
                metric = _burstiness_index(isis[into:outof], isis[outof:])
            elif i == 0: # was this the first one (and there weren't spikes after)?
                metric = _burstiness_index(isis[into:outof], isis[:into])
            else:
                prev_burst = inout_pairs[i - 1]
                metric = _burstiness_index(isis[into:outof], isis[prev_burst[1]:into])
        else:
            next_burst = inout_pairs[i + 1]
            metric = _burstiness_index(isis[into:outof], isis[outof:next_burst[0]])
        bursts.append((metric, into, outof))

    return bursts


def fit_prespike_time_constant(v, t, start, spike_time, dv_limit=-0.001, tau_limit=0.3):
    """Finds the dominant time constant of the pre-spike rise in voltage

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of voltage rise (seconds)
    spike_time : time of first spike (seconds)
    dv_limit : dV/dt cutoff (default -0.001)
        Shortens fit window if rate of voltage drop exceeds this limit
    tau_limit : upper bound for slow time constant (seconds, default 0.3)
        If the slower time constant of a double-exponential fit is twice that of the faster
        and exceeds this limit, the faster one will be considered the dominant one

    Returns
    -------
    tau : dominant time constant (seconds)
    """

    start_index = find_time_index(t, start)
    end_index = find_time_index(t, spike_time)
    if end_index <= start_index:
        raise FeatureError("Start for pre-spike time constant fit cannot be after the spike time.")

    v_slice = v[start_index:end_index]
    t_slice = t[start_index:end_index]

    # Solve linear version with single exponential first to guess at the time constant
    y0 = v_slice.max() + 5e-6 # set y0 slightly above v_slice maximum
    y = -v_slice + y0
    y = np.log(y)

    dy = calculate_dvdt(y, t_slice, filter=1.0)

    # End the fit interval if the voltage starts dropping
    new_end_indexes = np.flatnonzero(dy <= dv_limit)
    cross_limit = 0.0005 # sec
    if not new_end_indexes.size or t_slice[new_end_indexes[0]] - t_slice[0] < cross_limit:
        # either never crosses or crosses too early
        new_end_index = len(v_slice)
    else:
        new_end_index = new_end_indexes[0]

    K, A_log = np.polyfit(t_slice[:new_end_index] - t_slice[0], y[:new_end_index], 1)
    A = np.exp(A_log)

    dbl_exp_y0 = partial(_dbl_exp_fit, y0)
    try:
        popt, pcov = curve_fit(dbl_exp_y0, t_slice - t_slice[0], v_slice, p0=(-A / 2.0, -1.0 / K, -A / 2.0, -1.0 / K))
    except RuntimeError:
        # Fall back to single fit
        tau = -1.0 / K
        return tau

    # Find dominant time constant
    if popt[1] < popt[3]:
        faster_weight, faster_tau, slower_weight, slower_tau = popt
    else:
        slower_weight, slower_tau, faster_weight, faster_tau = popt

    # These are all empirical values
    if np.abs(faster_weight) > np.abs(slower_weight):
        tau = faster_tau
    elif (slower_tau - faster_tau) / slower_tau <= 0.1: # close enough; just use slower
        tau = slower_tau
    elif slower_tau > tau_limit and slower_weight / faster_weight < 2.0:
        tau = faster_tau
    else:
        tau = slower_tau

    return tau


def estimate_adjusted_detection_parameters(v_set, t_set, interval_start, interval_end, filter=10):
    """
    Estimate adjusted values for spike detection by analyzing a period when the voltage
    changes quickly but passively (due to strong current stimulation), which can result
    in spurious spike detection results.

    Parameters
    ----------
    v_set : list of numpy arrays of voltage time series in mV
    t_set : list of numpy arrays of times in seconds
    interval_start : start of analysis interval (sec)
    interval_end : end of analysis interval (sec)

    Returns
    -------
    new_dv_cutoff : adjusted dv/dt cutoff (V/s)
    new_thresh_frac : adjusted fraction of avg upstroke to find threshold
    """
    
    if type(v_set) is not list:
        v_set = list(v_set)

    if type(t_set) is not list:
        t_set = list(t_set)

    if len(v_set) != len(t_set):
        raise FeatureError("t_set and v_set must be lists of equal size")

    if len(v_set) == 0:
        raise FeatureError("t_set and v_set are empty")

    start_index = find_time_index(t_set[0], interval_start)
    end_index = find_time_index(t_set[0], interval_end)

    maxes = []
    ends = []
    dv_set = []
    for v, t in zip(v_set, t_set):
        dv = calculate_dvdt(v, t, filter)
        dv_set.append(dv)
        maxes.append(dv[start_index:end_index].max())
        ends.append(dv[end_index])

    maxes = np.array(maxes)
    ends = np.array(ends)

    cutoff_adj_factor = 1.1
    thresh_frac_adj_factor = 1.2

    new_dv_cutoff = np.median(maxes) * cutoff_adj_factor
    min_thresh = np.median(ends) * thresh_frac_adj_factor

    all_upstrokes = np.array([])
    for v, t, dv in zip(v_set, t_set, dv_set):
        putative_spikes = detect_putative_spikes(v, t, dv_cutoff=new_dv_cutoff, filter=filter)
        peaks = find_peak_indexes(v, t, putative_spikes)
        putative_spikes, peaks = filter_putative_spikes(v, t, putative_spikes, peaks)
        upstrokes = find_upstroke_indexes(v, t, putative_spikes, peaks, dvdt=dv)
        if upstrokes.size:
            all_upstrokes = np.append(all_upstrokes, dv[upstrokes])
    new_thresh_frac = min_thresh / all_upstrokes.mean()

    return new_dv_cutoff, new_thresh_frac


def _score_burst_set(bursts, isis, delta_t, c_n=0.1, c_tx=0.01):
    in_burst = np.zeros_like(isis, dtype=bool)
    for b in bursts:
        in_burst[b[0]:b[1]] = True

    # If all ISIs are part of a burst, give it a bad score
    if len(isis[~in_burst]) == 0:
        return [-1e12] * len(bursts)

    delta_frac = delta_t / isis

    scores = []
    for b in bursts:
        score = _burstiness_index(isis[b[0]:b[1]], isis[~in_burst]) # base score
        if b[1] < len(delta_t):
            score -= c_tx * (1. / (delta_frac[b[1]])) # cost for starting a burst
        if b[0] > 0:
            score -= c_tx * (1. / delta_frac[b[0] - 1]) # cost for ending a burst
        score -= c_n * (b[1] - b[0] - 1) # cost for extending a burst
        scores.append(score)

    return scores


def _burstiness_index(in_burst_isis, out_burst_isis):
    burst_rate = 1. / in_burst_isis.min()
    out_rate = 1. / out_burst_isis.min()
    return (burst_rate - out_rate) / (burst_rate + out_rate)


def _exp_curve(x, a, inv_tau, y0):
    return y0 + a * np.exp(-inv_tau * x)


def _dbl_exp_fit(y0, x, A1, tau1, A2, tau2):
    penalty = 0
    if tau1 < 0 or tau2 < 0:
        penalty = 1e6
    return y0 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + penalty


class FeatureError(Exception):
    """Generic Python-exception-derived object raised by feature detection functions."""
    pass
