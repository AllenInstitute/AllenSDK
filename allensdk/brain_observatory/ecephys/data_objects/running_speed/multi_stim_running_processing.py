from typing import Tuple
import numpy as np
import pandas as pd
from allensdk.brain_observatory import sync_utilities

from allensdk.brain_observatory.behavior.data_objects.\
    running_speed.running_processing import (
        get_running_df
    )


def _extract_dx_info(
        frame_times: np.ndarray,
        start_index: int,
        end_index: int,
        pkl_path: str,
        zscore_threshold: float = 10.0,
        use_lowpass_filter: bool = True
) -> pd.core.frame.DataFrame:
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
    zscore_threshold: float
        The threshold to use for removing outlier
        running speeds which might be noise and not true signal
    use_lowpass_filter: bool
        whther or not to apply a low pass filter to the
        running speed results

    Returns
    -------
    pd.DataFrame

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
                    theoretically be 5V (after crossing
                    5V goes to 0V, or vice versa). In
                    practice the encoder does not always
                    reach this value before wrapping, which can cause
                    transient spikes in speed at the voltage "wraps".
                "frame_time": list of the vsync times
                "dx": angular change, computed during data collection
            The raw data are provided so that the user may compute
            their own speed from source, if desired.

    """

    stim_file = pd.read_pickle(pkl_path)
    frame_times = frame_times[start_index:end_index]

    # occasionally an extra set of frame times are acquired
    # after the rest of the signals. We detect and remove these
    frame_times = sync_utilities.trim_discontiguous_times(frame_times)

    velocities = get_running_df(
                    stim_file,
                    frame_times,
                    use_lowpass_filter,
                    zscore_threshold
    )

    return velocities


def _get_behavior_frame_count(
    pkl_file_path: str
) -> int:
    """
    Get the number of frames in a behavior pickle file

    Parameters
    ----------
    pkl_file_path: string
        A path to a behavior pickle file
    """
    data = pd.read_pickle(pkl_file_path)

    return len(data["items"]["behavior"]['intervalsms']) + 1


def _get_frame_count(
    pkl_file_path: str
) -> int:
    """
    Get the number of frames in a mapping or replay pickle file

    Parameters
    ----------
    pkl_file_path: string
        A path to a mapping or replay pickle file
    """

    data = pd.read_pickle(pkl_file_path)

    return len(data['intervalsms']) + 1


def _get_frame_counts(
    behavior_pkl_path: str,
    mapping_pkl_path: str,
    replay_pkl_path: str
) -> Tuple[int, int, int]:
    """
    Get the number of frames for each stimulus

    Parameters
    ----------
    behavior_pkl_path: str
        path to behavior pickle file
    mapping_pkl_path: str
        path to mapping pickle file
    replay_pkl_path: str
        path to replay pickle file

    Return
    ------
    frame_counts: Tuple[int, int, int]
        (n_behavior, n_mapping, n_replay)
    """

    behavior_frame_count = _get_behavior_frame_count(
        pkl_file_path=behavior_pkl_path
    )

    mapping_frame_count = _get_frame_count(
        pkl_file_path=mapping_pkl_path
    )

    replay_frames_count = _get_frame_count(
        pkl_file_path=replay_pkl_path
    )

    return (behavior_frame_count,
            mapping_frame_count,
            replay_frames_count)
