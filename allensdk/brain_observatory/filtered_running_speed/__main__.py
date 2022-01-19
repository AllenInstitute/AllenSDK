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
from typing import Union, Iterable, Optional

from allensdk.brain_observatory.filtered_running_speed.stim_file import (
    CamStimOnePickleStimFile,
    BehaviorPickleFile,
    ReplayPickleFile
)

DEGREES_TO_RADIANS = np.pi / 180.0
INDEX_TO_BEHAVIOR = 0
INDEX_TO_MAPPING = 1
INDEX_TO_REPLAY = 2
DEFAULT_SAMPLING_FREQUENCY = 60
CRITICAL_FREQUENCY = 4
FILTER_ORDER = 3
DEFAULT_ZSCORE_THRESHOLD = 10.0
USE_LOWPASS_FILTER = True
WHEEL_DIAMETER_IN_INCHES = 6.5
START_FRAME = 0

# in cm/in
WHEEL_DIAMETER = WHEEL_DIAMETER_IN_INCHES * 2.54


def check_encoder(parent, key):
    """
    checks if encoder exists
    ---------
    parent: dict
        stim file data
    key: string
        A key to check
    Returns
    -------
    boolean
        True if encoder exists in stim data
    """
    result = True

    if len(parent["encoders"]) != 1:
        result = False

    elif key not in parent["encoders"][0]:
        result = False

    elif len(parent["encoders"][0][key]) == 0:
        result = False

    return result


def calc_deriv(x, time):
    """
    Calculate the derivative
    ---------
    x: np.ndarray
        data values
    time: np.ndarray
        time values
    Returns
    -------
    np.ndarray
        The derivative
    """
    dx = np.diff(x, prepend=np.nan)
    dt = np.diff(time, prepend=np.nan)

    return dx / dt


def _zscore_threshold_1d(
            data: np.ndarray,
            threshold: float = 5.0
        ) -> np.ndarray:
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

    running_radius = 0.5 * (
        # assume the animal runs at 2/3 the distance from the wheel center
        2.0 * WHEEL_DIAMETER / 3.0)
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
    dx_raw,
    frame_times,
    vsig,
    vin,
    start_index,
    end_index,
    lowpass: bool = True,
    zscore_threshold=DEFAULT_ZSCORE_THRESHOLD,
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
    dx_raw
        Deserialized 'pkl' file dx_deg
    frame_times: np.ndarray (1d)
        Timestamps for running data measurements
    vsig: np.ndarray (1d),
        raw analog data for encoder
    vin: np.ndarray (1d),
        input voltage to the encoder
    start_index: int,
        the start index of the stimulus in reference to the full vsync times
    end_index: int,
        the end index of the stimulus in reference to the full vsync times
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
            "velocity": computed running speed
            "net_rotation": dx in radians
            "frame_indexes": frame indexes into the raw frame time,
            "frame_times": the frame times for the timestamp data

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
    angular_speed = calc_deriv(
        angular_change,
        frame_times
    )  # speed in radians/s

    linear_speed = deg_to_dist(angular_speed)

    # Artifact correction to speed data
    wrap_corrected_linear_speed = _clip_speed_wraps(
        linear_speed, frame_times, np.concatenate([pos_wraps, neg_wraps]),
        t_span=0.25)

    outlier_corrected_linear_speed = _zscore_threshold_1d(
        wrap_corrected_linear_speed, threshold=zscore_threshold)

    # Final filtering (optional) for smoothing out the speed data
    if lowpass:
        polynomial_b, polynomial_a = signal.butter(
            FILTER_ORDER,
            Wn=CRITICAL_FREQUENCY,
            fs=DEFAULT_SAMPLING_FREQUENCY,
            btype="lowpass"
        )

        outlier_corrected_linear_speed = signal.filtfilt(
            polynomial_b,
            polynomial_a,
            np.nan_to_num(outlier_corrected_linear_speed)
        )

    dx_rad = degrees_to_radians(dx_raw)

    frame_indexes = list(range(start_index, end_index))

    df = pd.DataFrame(
        {
            "velocity": outlier_corrected_linear_speed[:len(frame_times)],
            "net_rotation": dx_rad[:len(frame_times)],
            "frame_indexes": frame_indexes,
            "frame_times": frame_times
        }
    )

    # due to an acquisition bug (the buffer of raw orientations may be updated
    # more slowly than it is read, leading to a 0 value for the change in
    # orientation over an interval) there may be exact zeros in the velocity.
    df = df[~(np.isclose(df["net_rotation"], 0.0))]

    return df


def match_timestamps(num_raw_timestamps, signal, label):
    """
    Sometimes the timestamp has one fewer value than
    the signal does, if this is the case, we can remove the extra value
    from the end of the signal

    Argument defaults and implementation suggestion via @dougo

    Parameters
    ----------
    num_raw_timestamps: list
        A list of timestamps
    signal: list
        A list of signal values
    label: string
        Name of the signal

    Returns
    -------
    list
        An adjusted signal to match the timestamp's length
    """

    if num_raw_timestamps == (len(signal) - 1):
        signal = signal[0:len(signal) - 1]

    if num_raw_timestamps != len(signal):
        raise ValueError(
            f"found {num_raw_timestamps} rising edges on the vsync line, "
            f"but found {len(signal)} samples for  {label}"
        )

    return signal


def extract_dx_info(
    frame_times,
    start_index,
    end_index,
    pkl_path
):
    """
    Extract all of the running speed data

    Parameters
    ----------
    frame_times: numpy.ndarray
        list of the vsync times
    start_index: int
        Index to the first frame of the stimulus
    end_index: int
       Index to the last frame of the stimulus
    pkl_path: string
        Path to the stimulus pickle file

    Returns
    -------
    list[pd.DataFrame, pd.DataFrame]
        the velocity data and the raw data

    Notes
    -------
            velocity pd.DataFrame:
                columns:
                    "velocity": computed running speed
                    "net_rotation": dx in radians
                    "frame_indexes": frame indexes into
                        the full vsync times list

            raw data pd.DataFrame:
                Dataframe with an index of timestamps and the following
                columns:
                    "vsig": voltage signal from the encoder
                    "vin": the theoretical maximum voltage that the encoder
                        will reach prior to "wrapping". This should
                        theoretically be 5V (after crossing 5V goes to 0V, or
                        vice versa). In practice the encoder does not always
                        reach this value before wrapping, which can cause
                        transient spikes in speed at the voltage "wraps".
                    "frame_time": list of the vsync times
                    "dx": angular change, computed during data collection
                The raw data are provided so that the user may compute their
                own speed from source, if desired.

    """

    stim_file = pd.read_pickle(pkl_path)

    frame_times = frame_times[start_index:end_index]

    # occasionally an extra set of frame times are acquired after the rest of
    # the signals. We detect and remove these
    frame_times = sync_utilities.trim_discontiguous_times(frame_times)
    num_raw_timestamps = len(frame_times)

    dx_deg = running_from_stim_file(stim_file, "dx", num_raw_timestamps)
    vsig = running_from_stim_file(stim_file, "vsig", num_raw_timestamps)
    vin = running_from_stim_file(stim_file, "vin", num_raw_timestamps)

    dx_deg = match_timestamps(num_raw_timestamps, dx_deg, 'dx')
    vsig = match_timestamps(num_raw_timestamps, vsig, 'vsig')
    vin = match_timestamps(num_raw_timestamps, vin, 'vin')

    velocities = extract_running_speeds(
        dx_deg,
        frame_times,
        vsig,
        vin,
        start_index,
        end_index,
        USE_LOWPASS_FILTER,
        DEFAULT_ZSCORE_THRESHOLD,
    )

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx_deg}
    )

    return velocities, raw_data


def merge_dx_data(
        mapping_velocities,
        mapping_raw_data,
        behavior_velocities,
        behavior_raw_data,
        replay_velocities,
        replay_raw_data
):
    """
    Concatenate all of the running speed data

    Parameters
    ----------
    mapping_velocities: pandas.core.frame.DataFrame
        Velocity data from mapping stimulus
    mapping_raw_data: pandas.core.frame.DataFrame
        Raw data from mapping stimulus
    behavior_velocities: pandas.core.frame.DataFrame
       Velocity data from behavior stimulus
    behavior_raw_data: pandas.core.frame.DataFrame
         Raw data from behavior stimulus
    replay_velocities: pandas.core.frame.DataFrame
        Velocity data from replay stimulus
    replay_raw_data: pandas.core.frame.DataFrame
         Raw data from replay stimulus

    Returns
    -------
    list[pd.DataFrame, pd.DataFrame]
        concatenated velocity data, concatenated raw data
    """

    velocity = np.concatenate(
        (
            behavior_velocities['velocity'],
            mapping_velocities['velocity'],
            replay_velocities['velocity']),
        axis=None
    )

    net_rotation = np.concatenate(
        (
            behavior_velocities['net_rotation'],
            mapping_velocities['net_rotation'],
            replay_velocities['net_rotation']),
        axis=None
    )

    frame_indexes = np.concatenate(
        (
            behavior_velocities['frame_indexes'],
            mapping_velocities['frame_indexes'],
            replay_velocities['frame_indexes']),
        axis=None
    )

    frame_times = np.concatenate(
        (
            behavior_velocities['frame_times'],
            mapping_velocities['frame_times'],
            replay_velocities['frame_times']),
        axis=None
    )

    vsig = np.concatenate(
        (
            behavior_raw_data['vsig'],
            mapping_raw_data['vsig'],
            replay_raw_data['vsig']),
        axis=None
    )

    vin = np.concatenate(
        (
            behavior_raw_data['vin'],
            mapping_raw_data['vin'],
            replay_raw_data['vin']
        ),
        axis=None
    )

    raw_frame_times = np.concatenate(
        (
            behavior_raw_data['frame_time'],
            mapping_raw_data['frame_time'],
            replay_raw_data['frame_time']
        ),
        axis=None
    )

    dx_deg = np.concatenate(
        (
            behavior_raw_data['dx'],
            mapping_raw_data['dx'],
            replay_raw_data['dx']
        ),
        axis=None
    )

    velocities = pd.DataFrame(
        {
            "velocity": velocity,
            "net_rotation": net_rotation,
            "frame_indexes": frame_indexes,
            "frame_time": frame_times
        }
    )

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": raw_frame_times, "dx": dx_deg}
    )

    return velocities, raw_data


def process_single_simulus_experiment(pkl_path, sync_h5_path, output_path):
    """
    Process an experiment with a single simulus session

    Parameters
    ----------
    pkl_path: string
        A path to the stimulus pickle file
    sync_h5_path: string
        A path to the sync file
    output_path: string
        A path to the output file
    """

    start_index = START_FRAME

    sync_data = Dataset(sync_h5_path)

    frame_times = sync_data.get_edges(
        "rising", Dataset.FRAME_KEYS, units="seconds"
    )

    end_index = len(frame_times)

    velocities, raw_data = extract_dx_info(
        frame_times,
        start_index,
        end_index,
        pkl_path
    )

    store = pd.HDFStore(output_path)
    store.put("running_speed", velocities)
    store.put("raw_data", raw_data)
    store.close()


def process_multi_simulus_experiment(
    mapping_pkl_path,
    behavior_pkl_path,
    replay_pkl_path,
    sync_h5_path,
    output_path
):
    """
    Process an experiment with a three simulus sessions

    Parameters
    ----------
    mapping_pkl_path: string
        A path to the mapping stimulus pickle file
    behavior_pkl_path: string
        A path to the behavior stimulus pickle file
    replay_pkl_path: string
        A path to the replay stimulus pickle file
    sync_h5_path: string
        A path to the sync file
    output_path: string
        A path to the output file
    """

    sync_data = Dataset(sync_h5_path)
    mapping_data = CamStimOnePickleStimFile.factory(mapping_pkl_path)
    behavior_data = BehaviorPickleFile.factory(behavior_pkl_path)
    replay_data = ReplayPickleFile.factory(replay_pkl_path)

    frame_counts = [
        pkl.num_frames for pkl in (behavior_data, mapping_data, replay_data)
    ]

    behavior_frame_count = frame_counts[INDEX_TO_BEHAVIOR]
    mapping_frame_count = frame_counts[INDEX_TO_MAPPING]
    replay_frames_count = frame_counts[INDEX_TO_REPLAY]

    behavior_start = START_FRAME
    behavior_end = behavior_frame_count

    mapping_start = behavior_end
    mapping_end = mapping_start + mapping_frame_count

    replay_start = mapping_end
    replay_end = replay_start + replay_frames_count

    # Why the rising edge? See Sweepstim.update in camstim. This method does:
    # 1. updates the stimuli
    # 2. updates the "items", causing a running speed sample to be acquired
    # 3. sets the vsync line high
    # 4. flips the buffer
    frame_times = sync_data.get_edges(
        "rising", Dataset.FRAME_KEYS, units="seconds"
    )

    behavior_velocities, behavior_raw_data = extract_dx_info(
        frame_times,
        behavior_start,
        behavior_end,
        behavior_pkl_path
    )

    mapping_velocities, mapping_raw_data = extract_dx_info(
        frame_times,
        mapping_start,
        mapping_end,
        mapping_pkl_path
    )

    replay_velocities, replay_raw_data = extract_dx_info(
        frame_times,
        replay_start,
        replay_end,
        replay_pkl_path
    )

    velocities, raw_data = merge_dx_data(
        mapping_velocities,
        mapping_raw_data,
        behavior_velocities,
        behavior_raw_data,
        replay_velocities,
        replay_raw_data
    )

    store = pd.HDFStore(output_path)
    store.put("running_speed", velocities)
    store.put("raw_data", raw_data)
    store.close()


def main(
        mapping_pkl_path,
        behavior_pkl_path,
        replay_pkl_path,
        sync_h5_path,
        output_path,
        wheel_radius,
        subject_position,
        use_median_duration,
        **kwargs
):
    print('running...')

    if mapping_pkl_path is not None:
        if behavior_pkl_path is not None and replay_pkl_path is not None:
            process_multi_simulus_experiment(
                mapping_pkl_path,
                behavior_pkl_path,
                replay_pkl_path,
                sync_h5_path,
                output_path
            )
        else:
            process_single_simulus_experiment(
                mapping_pkl_path,
                sync_h5_path,
                output_path
            )
    else:
        raise ValueError('Mapping pickle file has not been set')

    return {"output_path": output_path}


if __name__ == "__main__":
    mod = ArgSchemaParserPlus(
        schema_type=InputParameters, output_schema_type=OutputParameters
    )

    output = main(**mod.args)
    write_or_print_outputs(data=output, parser=mod)
