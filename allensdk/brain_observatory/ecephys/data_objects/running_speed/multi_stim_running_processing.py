from typing import Tuple
import numpy as np
import pandas as pd
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset

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
    velocities: pd.DataFrame

    Notes
    -----
    see allensdk.brain_observatory.behavior.data_objects.\
        running_speed.running_processing.get_running_df

    for detailed contents of output dataframe

    frame_times will be longer than the data arrays in the
    pickle file because frame_times encompasses all of the
    frames in the sync file, while the pickle file comprises
    just one of the stimulus files associated with this
    session. start_index and end_index are meant to trim
    the frame_times down to match the size of the data arrays
    stored in the pickle file.
    """

    stim_file = pd.read_pickle(pkl_path)
    frame_times = frame_times[start_index:end_index]

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


def _get_frame_times(
    sync_path: str
) -> np.ndarray:
    """
    Get the vsync frame times

    Parameters
    ----------
    sync_path: str
        Path to sync file

    Returns
    -------
    timestamps: np.ndarray
        numpy array of timestamps
    """
    sync_data = SyncDataset(sync_path)

    return sync_data.get_edges(
        "rising", SyncDataset.FRAME_KEYS, units="seconds"
    )


def _get_stimulus_starts_and_ends(
        behavior_pkl_path: str,
        mapping_pkl_path: str,
        replay_pkl_path: str,
        behavior_start_frame: int
) -> Tuple[int, int, int, int]:
    """
    Get the start and stop frame indexes for each stimulus

    Parameters
    ----------
    behavior_pkl_path: str
        path to behavior pickle file
    mapping_pkl_path: str
        path to mapping pickle file
    replay_pkl_path: str
        path to replay pickle file
    behavior_start_frame: int
        the frame at which behavior data is meant to start
        (this is effectively a truncation of the behavior
        frames, trimming off the first behavior_start_frame
        frames)

    Returns
    -------
    transition_frames: Tuple[int, int, int ,int]
        behavior start frame
        mapping start frame
        replay start frame
        replay end frame

    """

    (
        behavior_frame_count,
        mapping_frame_count,
        replay_frames_count
    ) = _get_frame_counts(
                behavior_pkl_path=behavior_pkl_path,
                mapping_pkl_path=mapping_pkl_path,
                replay_pkl_path=replay_pkl_path)

    behavior_start = behavior_start_frame
    mapping_start = behavior_frame_count
    replay_start = mapping_start + mapping_frame_count
    replay_end = replay_start + replay_frames_count

    if mapping_start <= behavior_start:
        raise RuntimeError(
             f"behavior_start_frame {behavior_start_frame} >= "
             f"mapping_start_frame {mapping_start}; "
             "unclear how to proceed")

    return (
        behavior_start,
        mapping_start,
        replay_start,
        replay_end
    )


def _merge_dx_data(
    mapping_velocities: pd.core.frame.DataFrame,
    behavior_velocities: pd.core.frame.DataFrame,
    replay_velocities: pd.core.frame.DataFrame,
    frame_times: np.ndarray,
    behavior_start_frame: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate all of the running speed data

    Parameters
    ----------
    mapping_velocities: pandas.core.frame.DataFrame
        Velocity data from mapping stimulus
    behavior_velocities: pandas.core.frame.DataFrame
       Velocity data from behavior stimulus
    replay_velocities: pandas.core.frame.DataFrame
        Velocity data from replay stimulus
    frame_times: numpy.ndarray
        list of the vsync times
    behavior_start_frame: int
        frame on which behavior data starts

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        concatenated velocity data, raw data
    """

    speed = np.concatenate(
        (
            behavior_velocities['speed'],
            mapping_velocities['speed'],
            replay_velocities['speed']),
        axis=None
        )

    dx = np.concatenate(
        (
            behavior_velocities['dx'],
            mapping_velocities['dx'],
            replay_velocities['dx']),
        axis=None
    )

    vsig = np.concatenate(
        (
            behavior_velocities['v_sig'],
            mapping_velocities['v_sig'],
            replay_velocities['v_sig']),
        axis=None
    )

    vin = np.concatenate(
        (
            behavior_velocities['v_in'],
            mapping_velocities['v_in'],
            replay_velocities['v_in']),
        axis=None
    )

    frame_indexes = list(
        range(behavior_start_frame, len(frame_times))
    )

    velocities = pd.DataFrame(
        {
            "velocity": speed,
            "net_rotation": dx,
            "frame_indexes": frame_indexes,
            "frame_time": frame_times
        }
    )

    # Warning - the 'isclose' line below needs to be refactored
    # as it exists in multiple places

    # due to an acquisition bug (the buffer of raw orientations
    # may be updated more slowly than it is read, leading to
    # a 0 value for the change in orientation over an interval)
    # there may be exact zeros in the velocity.
    velocities = velocities[~(np.isclose(velocities["net_rotation"], 0.0))]

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx}
    )

    return (velocities, raw_data)


def multi_stim_running_df_from_raw_data(
    sync_path: str,
    behavior_pkl_path: str,
    mapping_pkl_path: str,
    replay_pkl_path: str,
    use_lowpass_filter: bool,
    zscore_threshold: float,
    behavior_start_frame: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive running speed data frames from sync file and
    pickle files

    Parameters
    ----------
    sync_path: str
        The path to the sync file
    behavior_pkl_path: str
        path to behavior pickle file
    mapping_pkl_path: str
        path to mapping pickle file
    replay_pkl_path: str
        path to replay pickle file
    use_lowpass_filter: bool
        whther or not to apply a low pass filter to the
        running speed results
    zscore_threshold: float
        The threshold to use for removing outlier
        running speeds which might be noise and not true signal
    behavior_start_frame: int
        the frame on which behavior data starts

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        concatenated velocity data, raw data


    Notes
    -----
    velocities pd.DataFrame:
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

    (
        behavior_start,
        mapping_start,
        replay_start,
        replay_end
    ) = _get_stimulus_starts_and_ends(
            behavior_pkl_path=behavior_pkl_path,
            mapping_pkl_path=mapping_pkl_path,
            replay_pkl_path=replay_pkl_path,
            behavior_start_frame=behavior_start_frame)

    frame_times = _get_frame_times(
                      sync_path=sync_path)

    behavior_velocities = _extract_dx_info(
        frame_times=frame_times,
        start_index=behavior_start,
        end_index=mapping_start,
        pkl_path=behavior_pkl_path,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    mapping_velocities = _extract_dx_info(
        frame_times=frame_times,
        start_index=mapping_start,
        end_index=replay_start,
        pkl_path=mapping_pkl_path,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    replay_velocities = _extract_dx_info(
        frame_times=frame_times,
        start_index=replay_start,
        end_index=replay_end,
        pkl_path=replay_pkl_path,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    velocities, raw_data = _merge_dx_data(
        mapping_velocities=mapping_velocities,
        behavior_velocities=behavior_velocities,
        replay_velocities=replay_velocities,
        frame_times=frame_times,
        behavior_start_frame=behavior_start_frame
    )

    return (velocities, raw_data)
