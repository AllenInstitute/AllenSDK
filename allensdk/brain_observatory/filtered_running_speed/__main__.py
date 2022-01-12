import warnings

import numpy as np
import pandas as pd
import scipy.signal as signal

from scipy.stats import zscore

from allensdk.brain_observatory import sync_utilities
from allensdk.brain_observatory.argschema_utilities import \
    ArgSchemaParserPlus, \
    write_or_print_outputs
from allensdk.brain_observatory.sync_dataset import Dataset
from ._schemas import InputParameters, OutputParameters
from typing import List, Union, Tuple, Iterable, Optional

from ecephys_etl.data_extractors.stim_file import (
    CamStimOnePickleStimFile,
    BehaviorPickleFile,
    ReplayPickleFile
)

DEGREES_TO_RADIANS = np.pi / 180.0
INDEX_TO_BEHAVIOR = 0
INDEX_TO_MAPPING = 1
INDEX_TO_REPLAY = 2

def check_encoder(parent, key):
    if len(parent["encoders"]) != 1:
        return False
    if key not in parent["encoders"][0]:
        return False
    if len(parent["encoders"][0][key]) == 0:
        return False
    return True

def calc_deriv(x, time):
    dx = np.diff(x, prepend=np.nan)
    dt = np.diff(time, prepend=np.nan)
    return dx / dt

def _zscore_threshold_1d(data: np.ndarray,
                         threshold: float = 5.0) -> np.ndarray:
    """
    Replace values in 1d array `data` that exceed `threshold` number
    of SDs from the mean with NaN.
    Parameters
    ---------
    data: np.ndarray
        1d np array of values
    threshold: float (default=5.0)
        Z-score threshold to replace with NaN.
    Returns
    -------
    np.ndarray (1d)
        A copy of `data` with values exceeding `threshold` SDs from
        the mean replaced with NaN.
    """
    corrected_data = data.copy().astype("float")
    scores = zscore(data, nan_policy="omit")
    # Suppress warnings when comparing to nan values to reduce noise
    with np.errstate(invalid='ignore'):
        corrected_data[np.abs(scores) > threshold] = np.nan
    return corrected_data

def _local_boundaries(time, index, span: float = 0.25) -> tuple:
    """
    Given a 1d array of monotonically increasing timestamps, and a
    point in that array (`index`), compute the indices that form the
    inclusive boundary around `index` for timespan `span`.

    Values in `time` must monotonically increase. Flat lines (same value
    multiple times) are OK. The neighborhood may terminate around the
    index if the `span` is too small for the sampling rate. A warning
    will be raised in this case.

    Returns
    -------
    Tuple
        Tuple of corresponding to the start, end indices that bound
        a time span of length `span` (maximally)

    E.g.
    ```
    time = np.array([0, 1, 1.5, 2, 2.2, 2.5, 3, 3.5])
    _local_boundary(time, 3, 1.0)
    >>> (1, 6)
    ```
    """
    if np.diff(time[~np.isnan(time)]).min() < 0:
        raise ValueError("Data do not monotonically increase. This probably "
                         "means there is an error in your time series.")
    t_val = time[index]
    max_val = t_val + abs(span)
    min_val = t_val - abs(span)
    eligible_indices = np.nonzero((time <= max_val) & (time >= min_val))[0]
    max_ix = eligible_indices.max()
    min_ix = eligible_indices.min()
    if (min_ix == index) or (max_ix == index):
        warnings.warn("Unable to find two data points around index "
                      f"for span={span} that do not include the index. "
                      "This could mean that your time span is too small for "
                      "the time data sampling rate, the data are not "
                      "monotonically increasing, or that you are trying "
                      "to find a neighborhood at the beginning/end of the "
                      "data stream.")
    return min_ix, max_ix

def _clip_speed_wraps(speed, time, wrap_indices, t_span: float = 0.25):
    """
    Correct for artifacts at the voltage 'wraps'. Sometimes there are
    transient spikes in speed at the 'wrap' points. This doesn't make
    sense since speed on a running wheel should be a smoothly varying
    function. Take the neighborhood of values in +/- `t_span` seconds
    around wrap points, and clip the value at the wrap point
    such that it does not exceed the min/max values in the neighborhood.
    """
    corrected_speed = speed.copy()
    for wrap in wrap_indices:
        start_ix, end_ix = _local_boundaries(time, wrap, t_span)
        local_slice = np.concatenate(       # Remove the wrap point
            (speed[start_ix:wrap], speed[wrap+1:end_ix+1]))
        corrected_speed[wrap] = np.clip(
            speed[wrap], np.nanmin(local_slice), np.nanmax(local_slice))
    return corrected_speed

def deg_to_dist(angular_speed: np.ndarray) -> np.ndarray:
    """
    Takes the angular speed (radians/s) at each step in radians, and
    computes the linear speed in cm/s.

    Parameters
    ----------
    angular_speed: np.ndarray (1d)
        1d array of angular speed in radians/s
    Returns
    -------
    np.ndarray (1d)
        Linear speed in cm/s at each time point.
    """
    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter, 2.54 = cm/in
    running_radius = 0.5 * (
        # assume the animal runs at 2/3 the distance from the wheel center
        2.0 * wheel_diameter / 3.0)
    running_speed_cm_per_sec = angular_speed * running_radius
    return running_speed_cm_per_sec

def running_from_stim_file(stim_file, key, expected_length):
    if "behavior" in stim_file["items"] and check_encoder(
            stim_file["items"]["behavior"], key
    ):
        return stim_file["items"]["behavior"]["encoders"][0][key][:]
    if "foraging" in stim_file["items"] and check_encoder(
            stim_file["items"]["foraging"], key
    ):
        return stim_file["items"]["foraging"]["encoders"][0][key][:]
    if key in stim_file:
        return stim_file[key][:]

    warnings.warn(f"unable to read {key} from this stimulus file")
    return np.ones(expected_length) * np.nan


def degrees_to_radians(degrees):
    return np.array(degrees) * DEGREES_TO_RADIANS


def angular_to_linear_velocity(angular_velocity, radius):
    return np.multiply(angular_velocity, radius)

def _angular_change(summed_voltage: np.ndarray,
                    vmax: Union[np.ndarray, float]) -> np.ndarray:
    """
    Compute the change in degrees in radians at each point from the
    summed voltage encoder data.

    Parameters
    ----------
    summed_voltage: 1d np.ndarray
        The "unwrapped" voltage signal from the encoder, cumulatively
        summed. See `_unwrap_voltage_signal`.
    vmax: 1d np.ndarray or float
        Either a constant float, or a 1d array (typically constant)
        of values. These values represent the theoretical max voltage
        value of the encoder. If an array, needs to be the same length
        as the summed_voltage array.
    Returns
    -------
    np.ndarray
        1d array of change in degrees in radians from each point
    """
    delta_theta = np.diff(summed_voltage, prepend=np.nan) / vmax * 2 * np.pi
    return delta_theta

def _unwrap_voltage_signal(
        vsig: Iterable,
        pos_wrap_ix: Iterable,
        neg_wrap_ix: Iterable,
        *,
        vmax: Optional[float] = None,
        max_threshold: float = 5.1,
        max_diff: float = 1.0) -> np.ndarray:
    """
    Calculate the change in voltage at each timestamp.
    'Unwraps' the
    voltage data coming from the encoder at the value `vmax`. If `vmax`
    is a float, use that value to 'wrap'. If it is None, then compute
    the maximum value from the observed voltage signal (`vsig`, as long
    as the maximum value is under the value of `max_threshold` (to
    account for possible outlier data/encoder errors).
    The reason is because the rotary encoder should theoretically wrap
    at 5V, but in practice does not always reach 5V before wrapping
    back to 0V. If it is assumed that the encoder wraps at 5V, but
    actually does not reach that voltage, then the computed running
    speed can be transiently higher at the timestamps of the signal
    'wraps'.

    Parameters
    ----------
    vsig: Iterable (array-like)
        The raw voltage data from the rotary encoder
    vmax: Optional[float] (default=None)
        The value at which, upon passing this threshold, the voltage
         "wraps" back to 0V on the encoder.
    max_threshold: float (default=5.1)
        The maximum threshold for the `vmax` value. Used only if
        `vmax` is `None`. To account for the possibility of outlier
        data/encoder errors, the computed `vmax` should not exceed
        this value.
    max_diff: float (default=1.0)
        The maximum voltage difference allowed between two adjacent
        points, after accounting for the voltage "wrap". Values
        exceeding this threshold will be set to np.nan.
    Returns
    -------
    np.ndarray
        1d np.ndarray of the "unwrapped" signal from `vsig`.
    """
    if not isinstance(vsig, np.ndarray):
        vsig = np.array(vsig)
    if vmax is None:
        vmax = vsig[vsig < max_threshold].max()
    unwrapped_diff = np.zeros(vsig.shape)
    vsig_last = _shift(vsig)
    if len(pos_wrap_ix):
        # positive wraps: subtract from the previous value and add vmax
        unwrapped_diff[pos_wrap_ix] = (
            (vsig[pos_wrap_ix] + vmax) - vsig_last[pos_wrap_ix])
    # negative: subtract vmax and the previous value
    if len(neg_wrap_ix):
        unwrapped_diff[neg_wrap_ix] = (
            vsig[neg_wrap_ix] - (vsig_last[neg_wrap_ix] + vmax))
    # Other indices, just compute straight diff from previous value
    wrap_ix = np.concatenate((pos_wrap_ix, neg_wrap_ix))
    other_ix = np.array(list(set(range(len(vsig_last))).difference(wrap_ix)))
    unwrapped_diff[other_ix] = vsig[other_ix] - vsig_last[other_ix]
    # Correct for wrap artifacts based on allowed `max_diff` value
    # (fill with nan)
    # Suppress warnings when comparing with nan values to reduce noise
    with np.errstate(invalid='ignore'):
        unwrapped_diff = np.where(
            np.abs(unwrapped_diff) <= max_diff, unwrapped_diff, np.nan)
    # Get nan indices to propogate to the cumulative sum (otherwise
    # treated as 0)
    unwrapped_nans = np.array(np.isnan(unwrapped_diff)).nonzero()
    summed_diff = np.nancumsum(unwrapped_diff) + vsig[0]    # Add the baseline
    summed_diff[unwrapped_nans] = np.nan
    return summed_diff

def _shift(
        arr: Iterable,
        periods: int = 1,
        fill_value: float = np.nan) -> np.ndarray:
    """
    Shift index of an iterable (array-like) by desired number of
    periods with an optional fill value (default = NaN).

    Parameters
    ----------
    arr: Iterable (array-like)
        Iterable containing numeric data. If int, will be converted to
        float in returned object.
    periods: int (default=1)
        The number of elements to shift.
    fill_value: float (default=np.nan)
        The value to fill at the beginning of the shifted array
    Returns
    -------
    np.ndarray (1d)
        Copy of input object as a 1d array, shifted.
    """
    if periods <= 0:
        raise ValueError("Can only shift for periods > 0.")
    if fill_value is None:
        fill_value = np.nan
    if isinstance(fill_value, float):
        # Circumvent issue if int-like array with np.nan as fill
        shifted = np.roll(arr, periods).astype(float)
    else:
        shifted = np.roll(arr, periods)
    shifted[:periods] = fill_value
    return shifted


def _identify_wraps(vsig: Iterable, *,
                    min_threshold: float = 1.5,
                    max_threshold: float = 3.5):
    """
    Identify "wraps" in the voltage signal. In practice, this is when
    the encoder voltage signal crosses 5V and wraps to 0V, or
    vice-versa.

    Argument defaults and implementation suggestion via @dougo

    Parameters
    ----------
    vsig: Iterable (array-like)
        1d array-like iterable of voltage signal
    min_threshold: float (default=1.5)
        The min_threshold value that must be crossed to be considered
        a possible wrapping point.
    max_threshold: float (default=3.5)
        The max threshold value that must be crossed to be considered
        a possible wrapping point.

    Returns
    -------
    Tuple
        Tuple of ([indices of positive wraps], [indices of negative wraps])
    """
    # Compare against previous value
    shifted_vsig = _shift(vsig)
    if not isinstance(vsig, np.ndarray):
        vsig = np.array(vsig)
    # Suppress warnings for when comparing to nan values
    with np.errstate(invalid='ignore'):
        pos_wraps = np.asarray(
            np.logical_and(vsig < min_threshold, shifted_vsig > max_threshold)
            ).nonzero()[0]
        neg_wraps = np.asarray(
            np.logical_and(vsig > max_threshold, shifted_vsig < min_threshold)
            ).nonzero()[0]

    return pos_wraps, neg_wraps

def extract_running_speeds(
    dx_raw, frame_times: np.ndarray, vsig, vin, wheel_radius, subject_position, use_median_duration, lowpass: bool = True, zscore_threshold=10.0
):
    """
    Given the dx_deg from the 'pkl' file object and a 1d
    array of timestamps, compute the running speed. Returns a
    dataframe with the raw voltage dx_deg as well as the computed speed
    at each timestamp. By default, the running speed is filtered with
    a 10 Hz Butterworth lowpass filter to remove artifacts caused by
    the rotary encoder.

    Parameters
    ----------
    dx_deg
        Deserialized 'behavior pkl' file dx_deg
    frame_times: np.ndarray (1d)
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
            "vsig": voltage signal from the encoder
            "vin": the theoretical maximum voltage that the encoder
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

    # if len(vin) > len(frame_times) + 1:
    #     error_string = ("length of vin ({}) cannot be longer than length of "
    #                     "frame_times ({}) + 1, they are off by {}").format(
    #         len(vin),
    #         len(frame_times),
    #         abs(len(vin) - len(frame_times))
    #     )
    #     raise ValueError(error_string)
    # if len(vin) == len(frame_times) + 1:
    #     warnings.warn(
    #         "frame_times array is 1 value shorter than encoder array. Last encoder "
    #         "value removed\n", UserWarning, stacklevel=1)
    #     vin = vin[:-1]
    #     vsig = vsig[:-1]

    # dx = 'd_theta' = angular change
    # There are some issues with angular change in the raw data so we
    # recompute this value
    # dx_raw = data["items"]["behavior"]["encoders"][0]["dx"]

    # Identify "wraps" in the voltage signal that need to be unwrapped
    # This is where the encoder switches from 0V to 5V or vice versa
    pos_wraps, neg_wraps = _identify_wraps(
        vsig, min_threshold=1.5, max_threshold=3.5)
    # Unwrap the voltage signal and apply correction for transient spikes
    unwrapped_vsig = _unwrap_voltage_signal(
        vsig, pos_wraps, neg_wraps, max_threshold=5.1, max_diff=1.0)
    angular_change_point = _angular_change(unwrapped_vsig, vin)
    angular_change = np.nancumsum(angular_change_point)
    # Add the nans back in (get turned to 0 in nancumsum)
    angular_change[np.isnan(angular_change_point)] = np.nan
    angular_speed = calc_deriv(angular_change, frame_times)  # speed in radians/s
    linear_speed = deg_to_dist(angular_speed)
    # Artifact correction to speed data
    wrap_corrected_linear_speed = _clip_speed_wraps(
        linear_speed, frame_times, np.concatenate([pos_wraps, neg_wraps]),
        t_span=0.25)


    outlier_corrected_linear_speed = _zscore_threshold_1d(
        wrap_corrected_linear_speed, threshold=zscore_threshold)

    # Final filtering (optional) for smoothing out the speed data
    if lowpass:
        b, a = signal.butter(3, Wn=4, fs=60, btype="lowpass")
        outlier_corrected_linear_speed = signal.filtfilt(
            b, a, np.nan_to_num(outlier_corrected_linear_speed))

    # return pd.DataFrame({
    #     'speed': outlier_corrected_linear_speed[:len(frame_times)],
    #     'dx': dx_raw[:len(frame_times)],
    #     'vsig': vsig[:len(frame_times)],
    #     'vin': vin[:len(frame_times)],
    # }, index=pd.Index(frame_times, name='timestamps'))

    dx_rad = degrees_to_radians(dx_raw)

    # durations = end_times - start_times
    # if use_median_duration:
    #     angular_velocity = dx_rad / np.median(durations)
    # else:
    #     angular_velocity = dx_rad / durations

    df = pd.DataFrame(
        {
            "velocity": outlier_corrected_linear_speed[:len(frame_times)],
            "net_rotation": dx_rad[:len(frame_times)],
        }
    )

    # due to an acquisition bug (the buffer of raw orientations may be updated
    # more slowly than it is read, leading to a 0 value for the change in
    # orientation over an interval) there may be exact zeros in the velocity.
    df = df[~(np.isclose(df["net_rotation"], 0.0))]

    return df

def old_extract_running_speeds(
        frame_times, dx_deg, vsig, vin, wheel_radius, subject_position,
        use_median_duration=False
):
    # the first interval does not have a known start time, so we can't compute
    # an average velocity from dx
    dx_rad = degrees_to_radians(dx_deg[1:])

    start_times = frame_times[:-1]
    end_times = frame_times[1:]

    durations = end_times - start_times
    if use_median_duration:
        angular_velocity = dx_rad / np.median(durations)
    else:
        angular_velocity = dx_rad / durations

    radius = wheel_radius * subject_position
    linear_velocity = angular_to_linear_velocity(angular_velocity, radius)

    df = pd.DataFrame(
        {
            "start_time": start_times,
            "end_time": end_times,
            "velocity": linear_velocity,
            "net_rotation": dx_rad,
        }
    )

    # due to an acquisition bug (the buffer of raw orientations may be updated
    # more slowly than it is read, leading to a 0 value for the change in
    # orientation over an interval) there may be exact zeros in the velocity.
    df = df[~(np.isclose(df["net_rotation"], 0.0))]

    return df

def extract_all_dx_info(behavior_pkl_path, mapping_pkl_path, replay_pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration, use_old_running_speed_method=True):


    behavior_pkl = pd.read_pickle(behavior_pkl_path)
    mapping_pkl = pd.read_pickle(mapping_pkl_path)
    replay_pkl = pd.read_pickle(replay_pkl_path)

    sync_dataset = Dataset(sync_h5_path)


    # print('pkl_path', pkl_path)

    # Why the rising edge? See Sweepstim.update in camstim. This method does:
    # 1. updates the stimuli
    # 2. updates the "items", causing a running speed sample to be acquired
    # 3. sets the vsync line high
    # 4. flips the buffer
    frame_times = sync_dataset.get_edges(
        "rising", Dataset.UPDATED_FRAME_KEYS, units="seconds"
    )

    # print('frame_times' , len(frame_times))

    # frame_times = frame_times[start_index:end_index]

    # occasionally an extra set of frame times are acquired after the rest of
    # the signals. We detect and remove these
    # frame_times = sync_utilities.trim_discontiguous_times(frame_times)
    num_raw_timestamps = len(frame_times)
    # num_raw_timestamps = 0

    # print('stim_file' , stimulus_pkl_path)

    behavior_dx_deg = running_from_stim_file(behavior_pkl, "dx", num_raw_timestamps)
    mapping_dx_deg = running_from_stim_file(mapping_pkl, "dx", num_raw_timestamps)
    replay_dx_deg = running_from_stim_file(replay_pkl, "dx", num_raw_timestamps)

    dx_deg = np.concatenate((behavior_dx_deg, mapping_dx_deg, replay_dx_deg), axis=None)

    #TODO take out
    # dx_deg = dx_deg[0:len(dx_deg) - 1]

    # print('sum2', len(dx_deg))
    # np.concatenate((a, b), axis=None)
    # np.concatenate((a, b), axis=None)

    # print('type', mapping_dx_deg)

    # dx_deg = behavior_dx_deg + mapping_dx_deg + replay_dx_deg

    # if num_raw_timestamps != len(dx_deg):
    #     raise ValueError(
    #         f"found {num_raw_timestamps} rising edges on the vsync line, "
    #         f"but only {len(dx_deg)} rotation samples"
    #     )
    # else:
    #     print('found', num_raw_timestamps, ' and ', len(dx_deg))

    behavior_vsig = running_from_stim_file(behavior_pkl, "vsig", num_raw_timestamps)
    behavior_vin = running_from_stim_file(behavior_pkl, "vin", num_raw_timestamps)

    mapping_vsig = running_from_stim_file(mapping_pkl, "vsig", num_raw_timestamps)
    mapping_vin = running_from_stim_file(mapping_pkl, "vin", num_raw_timestamps)

    replay_vsig = running_from_stim_file(replay_pkl, "vsig", num_raw_timestamps)
    replay_vin = running_from_stim_file(replay_pkl, "vin", num_raw_timestamps)

    vsig = np.concatenate((behavior_vsig, mapping_vsig, replay_vsig), axis=None)
    vin =  np.concatenate((behavior_vin, mapping_vin, replay_vin), axis=None)

    velocities = None

    if use_old_running_speed_method:
        velocities = old_extract_running_speeds(
            frame_times=frame_times,
            dx_deg=dx_deg,
            vsig=vsig,
            vin=vin,
            wheel_radius=wheel_radius,
            subject_position=subject_position,
            use_median_duration=use_median_duration
        )
    else:
        velocities = extract_running_speeds(dx_deg, frame_times, vsig, vin, wheel_radius, subject_position, use_median_duration, True, 10.0)



    # print('vsig', len(vsig))
    # print('vin', len(vin))
    # print('frame_times', len(frame_times))
    # print('dx_deg', len(dx_deg))

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx_deg}
    )

    return velocities, raw_data

def extract_dx_info(start_index, end_index, pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration, use_old_running_speed_method=False):
    stim_file = pd.read_pickle(pkl_path)
    sync_dataset = Dataset(sync_h5_path)



    # Why the rising edge? See Sweepstim.update in camstim. This method does:
    # 1. updates the stimuli
    # 2. updates the "items", causing a running speed sample to be acquired
    # 3. sets the vsync line high
    # 4. flips the buffer
    frame_times = sync_dataset.get_edges(
        "rising", Dataset.UPDATED_FRAME_KEYS, units="seconds"
    )

    # print('pkl_path', pkl_path)

    frame_times = frame_times[start_index:end_index]

    # occasionally an extra set of frame times are acquired after the rest of
    # the signals. We detect and remove these
    frame_times = sync_utilities.trim_discontiguous_times(frame_times)
    num_raw_timestamps = len(frame_times)

    # print('stim_file' , stimulus_pkl_path)

    dx_deg = running_from_stim_file(stim_file, "dx", num_raw_timestamps)

    if num_raw_timestamps == (len(dx_deg) - 1):
        dx_deg = dx_deg[0:len(dx_deg) - 1]


    if num_raw_timestamps != len(dx_deg):
        raise ValueError(
            f"found {num_raw_timestamps} rising edges on the vsync line, "
            f"but only {len(dx_deg)} rotation samples"
        )

    vsig = running_from_stim_file(stim_file, "vsig", num_raw_timestamps)
    vin = running_from_stim_file(stim_file, "vin", num_raw_timestamps)

    if num_raw_timestamps == (len(vsig) - 1):
        vsig = vsig[0:len(vsig) - 1]

    if num_raw_timestamps == (len(vin) - 1):
        vin = vin[0:len(vin) - 1]

    velocities = None

    if use_old_running_speed_method:
        velocities = old_extract_running_speeds(
            frame_times=frame_times,
            dx_deg=dx_deg,
            vsig=vsig,
            vin=vin,
            wheel_radius=wheel_radius,
            subject_position=subject_position,
            use_median_duration=use_median_duration
        )
    else:
        velocities = extract_running_speeds(dx_deg, frame_times, vsig, vin, wheel_radius, subject_position, use_median_duration, True, 10.0)

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx_deg}
    )

    return velocities, raw_data

def merge_dx_data(sync_h5_path, mapping_velocities, mapping_raw_data, behavior_velocities, behavior_raw_data, replay_velocities, replay_raw_data):

    # start_time = mapping_velocities['start_time'] + behavior_velocities['start_time'] + replay_velocities['start_time']
    # end_time = mapping_velocities['end_time'] + behavior_velocities['end_time'] + replay_velocities['end_time']
    velocity = np.concatenate((mapping_velocities['velocity'], behavior_velocities['velocity'], replay_velocities['velocity']), axis=None)
    dx_rad = np.concatenate((mapping_velocities['net_rotation'], behavior_velocities['net_rotation'], replay_velocities['net_rotation']), axis=None)

    vsig = np.concatenate((mapping_raw_data['vsig'], behavior_raw_data['vsig'], replay_raw_data['vsig']), axis=None)
    vin = np.concatenate((mapping_raw_data['vin'], behavior_raw_data['vin'], replay_raw_data['vin']), axis=None)
    frame_time = np.concatenate((mapping_raw_data['frame_time'], behavior_raw_data['frame_time'], replay_raw_data['frame_time']), axis=None)
    dx_deg = np.concatenate((mapping_raw_data['dx'], behavior_raw_data['dx'], replay_raw_data['dx']), axis=None)

    # frame_times

    sync_dataset = Dataset(sync_h5_path)

    frame_times = sync_dataset.get_edges(
        "rising", Dataset.UPDATED_FRAME_KEYS, units="seconds"
    )

    start_times = frame_times[:-1]
    end_times = frame_times[1:]

    velocities = pd.DataFrame(
        {
            # "start_time": start_times,
            # "end_time": end_times,
            "velocity": velocity,
            "net_rotation": dx_rad,
        }
    )

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_time, "dx": dx_deg}
    )

    return velocities, raw_data

def process_single_simulus_experiment(pkl_path, sync_h5_path, output_path, wheel_radius, subject_position, use_median_duration):
    start_index = 0

    sync_data = Dataset(sync_h5_path)

    frame_times = sync_data.get_edges(
        "rising", Dataset.UPDATED_FRAME_KEYS, units="seconds"
    )

    end_index = len(frame_times)

    velocities, raw_data = extract_dx_info(start_index, end_index, pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration)

    store = pd.HDFStore(output_path)
    store.put("running_speed", velocities)
    store.put("raw_data", raw_data)
    store.close()

def process_multi_simulus_experiment(mapping_pkl_path, behavior_pkl_path, replay_pkl_path, sync_h5_path, output_path, wheel_radius, subject_position, use_median_duration):
    sync_data = Dataset(sync_h5_path)
    mapping_data = CamStimOnePickleStimFile.factory(mapping_pkl_path)
    behavior_data = BehaviorPickleFile.factory(behavior_pkl_path)
    replay_data = ReplayPickleFile.factory(replay_pkl_path)

    frame_counts = [
        pkl.num_frames for pkl in (behavior_data, mapping_data, replay_data)
    ]

    # frame_offsets, stim_starts, stim_ends = (
    #     get_frame_offsets(sync_data, frame_counts)
    # )

    behavior_frame_count = frame_counts[INDEX_TO_BEHAVIOR]
    mapping_frame_count = frame_counts[INDEX_TO_MAPPING]
    replay_frames_count = frame_counts[INDEX_TO_REPLAY]

    behavior_start = 0 
    behavior_end = behavior_frame_count

    mapping_start = behavior_end
    mapping_end = mapping_start + mapping_frame_count

    replay_start = mapping_end
    replay_end = replay_start + replay_frames_count

    # extract_all_dx_info(behavior_pkl_path, mapping_pkl_path, replay_pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration)

    behavior_velocities, behavior_raw_data = extract_dx_info(behavior_start, behavior_end, behavior_pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration)

    mapping_velocities, mapping_raw_data = extract_dx_info(mapping_start, mapping_end, mapping_pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration)

    replay_velocities, replay_raw_data = extract_dx_info(replay_start, replay_end, replay_pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration)

    # mapping_velocities, mapping_raw_data = extract_dx_info(replay_start, replay_end, replay_pkl_path, sync_h5_path, wheel_radius, subject_position, use_median_duration)

    velocities, raw_data = merge_dx_data(sync_h5_path, mapping_velocities, mapping_raw_data, behavior_velocities, behavior_raw_data, replay_velocities, replay_raw_data)

    store = pd.HDFStore(output_path)
    store.put("running_speed", velocities)
    store.put("raw_data", raw_data)
    store.close()

def main(
        mapping_pkl_path, behavior_pkl_path, replay_pkl_path, sync_h5_path, output_path, wheel_radius,
        subject_position, use_median_duration, **kwargs
):
    print('running...   ', mapping_pkl_path)

    result = None

    if mapping_pkl_path is not None:
        if behavior_pkl_path is not None and replay_pkl_path is not None:
            process_multi_simulus_experiment(mapping_pkl_path, behavior_pkl_path, replay_pkl_path, sync_h5_path, output_path, wheel_radius, subject_position, use_median_duration)
        else: 
            process_single_simulus_experiment(mapping_pkl_path, sync_h5_path, output_path, wheel_radius, subject_position, use_median_duration)
    else:
        raise ValueError('Mapping pickle file has not been set')

    return {"output_path": output_path}

if __name__ == "__main__":
    mod = ArgSchemaParserPlus(
        schema_type=InputParameters, output_schema_type=OutputParameters
    )

    output = main(**mod.args)
    write_or_print_outputs(data=output, parser=mod)
