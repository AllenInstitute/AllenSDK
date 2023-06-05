from typing import Tuple, Dict, Union
import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.data_files.stimulus_file import (
    BehaviorStimulusFile,
    MappingStimulusFile,
    ReplayStimulusFile,
    _StimulusFile)

from allensdk.brain_observatory.sync_stim_aligner import (
    get_stim_timestamps_from_stimulus_blocks)

from allensdk.brain_observatory.behavior.data_objects.\
    running_speed.running_processing import (
        get_running_df
    )
 
from allensdk.brain_observatory import sync_dataset
import logging


def _extract_dx_info(
        frame_times: np.ndarray,
        stimulus_file: _StimulusFile,
        zscore_threshold: float = 10.0,
        use_lowpass_filter: bool = True
) -> pd.core.frame.DataFrame:
    """
    Extract all of the running speed data

    Parameters
    ----------
    frame_times: numpy.ndarray
        list of the vsync times corresponding to this
        stimulus block
    stimulus_file: _StimulusFile
        _StimulusFile object from which to get running data
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
    """

    stim_file = stimulus_file.data

    velocities = get_running_df(
                    stim_file,
                    frame_times,
                    use_lowpass_filter,
                    zscore_threshold
    )

    return velocities


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
        list of the vsync times for all three stimulus blocks
        together
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
            replay_velocities['speed']
            ),
        axis=None
        )

    dx = np.concatenate(
        (
            behavior_velocities['dx'],
            mapping_velocities['dx'],
            replay_velocities['dx']
            ),
        axis=None
    )

    vsig = np.concatenate(
        (
            behavior_velocities['v_sig'],
            mapping_velocities['v_sig'],
            replay_velocities['v_sig']
            ),
        axis=None
    )

    vin = np.concatenate(
        (
            behavior_velocities['v_in'],
            mapping_velocities['v_in'],
            replay_velocities['v_in']
            ),
        axis=None
    )

    frame_indexes = list(
        range(behavior_start_frame,
              behavior_start_frame+len(frame_times))
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
    # is it exists in multiple places

    # due to an acquisition bug (the buffer of raw orientations
    # may be updated more slowly than it is read, leading to
    # a 0 value for the change in orientation over an interval)
    # there may be exact zeros in the velocity.
    velocities = velocities[~(np.isclose(velocities["net_rotation"], 0.0))]

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx}
    )

    return (velocities, raw_data)

def _merge_dx_data_no_replay(
    mapping_velocities: pd.core.frame.DataFrame,
    behavior_velocities: pd.core.frame.DataFrame,
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
        list of the vsync times for all three stimulus blocks
        together
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
            ),
        axis=None
        )

    dx = np.concatenate(
        (
            behavior_velocities['dx'],
            mapping_velocities['dx'],
            ),
        axis=None
    )

    vsig = np.concatenate(
        (
            behavior_velocities['v_sig'],
            mapping_velocities['v_sig'],
            ),
        axis=None
    )

    vin = np.concatenate(
        (
            behavior_velocities['v_in'],
            mapping_velocities['v_in'],
            ),
        axis=None
    )

    frame_indexes = list(
        range(behavior_start_frame,
              behavior_start_frame+len(frame_times))
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
    # is it exists in multiple places

    # due to an acquisition bug (the buffer of raw orientations
    # may be updated more slowly than it is read, leading to
    # a 0 value for the change in orientation over an interval)
    # there may be exact zeros in the velocity.
    velocities = velocities[~(np.isclose(velocities["net_rotation"], 0.0))]

    raw_data = pd.DataFrame(
        {"vsig": vsig, "vin": vin, "frame_time": frame_times, "dx": dx}
    )

    return (velocities, raw_data)

def get_stim_starts_and_ends(
    sync_dataset: sync_dataset.Dataset, fallback_line: Union[int, str] = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Get stimulus presentation start and end times from a loaded session
    *.sync datset.

    Parameters
    ----------
    sync_dataset : Dataset
        A loaded *.sync file for a session (contains events from
        different data streams logged on a global time basis)
    fallback_line : Union[int, str], optional
        The sync dataset line label to use if named line labels could not
        be found, by default 5.

        For more details about line labels see:
        https://alleninstitute.sharepoint.com/:x:/s/Instrumentation/ES2bi1xJ3E9NupX-zQeXTlYBS2mVVySycfbCQhsD_jPMUw?e=Z9jCwH  # noqa: E501

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of numpy arrays containing
        (stimulus_start_times, stimulus_end_times) in seconds.
    """

    # Look for 'stim_running' line in sync dataset line labels
    stim_line: Union[int, str] = fallback_line
    for line in sync_dataset.line_labels:
        if line == 'stim_running':
            stim_line = line
            break
        if line == 'sweep':
            stim_line = line
            break

    if stim_line == fallback_line:
        logging.warning(
            f"Could not find 'stim_running' nor 'sweep' line labels in "
            f"sync dataset ({sync_dataset.dfile.filename}). Defaulting to "
            f"using fallback line label index ({fallback_line}) which "
            f"is not guaranteed to be correct!"
        )

    # 'stim_running'/'sweep' line is high while visual stimulus is being
    # displayed and low otherwise
    stim_starts = sync_dataset.get_rising_edges(stim_line, units='seconds')
    stim_ends = sync_dataset.get_falling_edges(stim_line, units='seconds')

    return stim_starts, stim_ends
def get_stim_epoch_vsync(sync: sync_dataset.Dataset):
    """
    gets the vsyncs for each stimulus in session
    """

    vsync_times = sync.get_falling_edges('vsync_stim', 'seconds')
    stim_starts, stim_ends = get_stim_starts_and_ends(sync)
    final_vsyncs = []

    for index_epoch, (start, end) in enumerate(zip(stim_starts, stim_ends)):
        epoch_vsyncs = vsync_times[(vsync_times >= start) & (vsync_times <= end)]
        final_vsyncs.append(epoch_vsyncs)

    return final_vsyncs

def get_stim_order(sync: sync_dataset.Dataset, pkl_file_list: list) -> np.array:
    """Gets the order that the stimuli were shown

    Parameters
    sync: Dataset
        A sync Datset object that allows pythonic access of *.sync file data
        containing global timing information for events and presented stimuli.
    pkl_file_list: list
        A list of the pickle files for the behavior and mapping stimuli

    Returns
    np.array
        Array with order of behavior and mapping stimuli
    """

    number_pkl_frames = np.array([pkl.num_frames for pkl in pkl_file_list])
    sync_stim_epoch_frames = get_stim_epoch_vsync(sync)
    sync_stim_epoch_frames_counts = [len(sync_epoch_frame) for sync_epoch_frame in sync_stim_epoch_frames]

    for sync_epoch_frames_count in sync_stim_epoch_frames_counts:
        if sync_epoch_frames_count - 1 in number_pkl_frames:
            index = sync_stim_epoch_frames_counts.index(sync_epoch_frames_count)
            sync_stim_epoch_frames_counts[index] = sync_epoch_frames_count - 1


    if len(sync_stim_epoch_frames_counts) > 2:
        for sync_stim_epoch_frame_count in sync_stim_epoch_frames_counts:
            if sync_stim_epoch_frame_count not in number_pkl_frames:
                sync_stim_epoch_frames_counts.remove(sync_stim_epoch_frame_count)

    print(sync_stim_epoch_frames_counts)

    if not all([pkl_count in sync_stim_epoch_frames_counts for pkl_count in number_pkl_frames]):
        raise ValueError('No match between the pkl frame counts and the sync frame counts')

    order = [np.where(number_pkl_frames == sync_count)[0][0] for sync_count in sync_stim_epoch_frames_counts]
    print(order)

    start_times = [pkl.start_time for pkl in pkl_file_list]
    if start_times[0] < start_times[1]:
        order_time = [0, 1]
    else:
        order_time = [1, 0]

    if not all([x == y for x, y in zip(order, order_time)]):
        raise ValueError('Order does not match using start times and number of frames')

    return order

def multi_stim_running_df_from_raw_data(
    sync_path: str,
    behavior_stimulus_file: BehaviorStimulusFile,
    mapping_stimulus_file: MappingStimulusFile,
    replay_stimulus_file: ReplayStimulusFile,
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
    behavior_stimulus_file: BehaviorStimulusFile
        stimulus file for the behavior stimulus block
    mapping_stimulus_file: MappingStimulusFile
        stimulus file for the mapping stimulus block
    replay_stimulus_file: ReplayStimulusFile
        stimulus file for the replay stimulus block
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

    timestamp_results = get_stim_timestamps_from_stimulus_blocks(
                              stimulus_files=[behavior_stimulus_file,
                                              mapping_stimulus_file,
                                              replay_stimulus_file],
                              sync_file=sync_path,
                              raw_frame_time_lines=['frames',
                                                    'stim_vsync',
                                                    'vsync_stim'],
                              raw_frame_time_direction='rising',
                              frame_count_tolerance=0.0)

    behavior_timestamps = timestamp_results["timestamps"][0]
    mapping_timestamps = timestamp_results["timestamps"][1]
    replay_timestamps = timestamp_results["timestamps"][2]

    start_frame = timestamp_results["start_frames"][0]

    behavior_velocities = _extract_dx_info(
        frame_times=behavior_timestamps,
        stimulus_file=behavior_stimulus_file,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    mapping_velocities = _extract_dx_info(
        frame_times=mapping_timestamps,
        stimulus_file=mapping_stimulus_file,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    replay_velocities = _extract_dx_info(
        frame_times=replay_timestamps,
        stimulus_file=replay_stimulus_file,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    all_frame_times = np.concatenate(
            [behavior_timestamps,
             mapping_timestamps,
             replay_timestamps])

    velocities, raw_data = _merge_dx_data(
        mapping_velocities=mapping_velocities,
        behavior_velocities=behavior_velocities,
        replay_velocities=replay_velocities,
        frame_times=all_frame_times,
        behavior_start_frame=start_frame
    )

    return (velocities, raw_data)


def _get_multi_stim_running_df(
        sync_path: str,
        behavior_stimulus_file: BehaviorStimulusFile,
        mapping_stimulus_file: MappingStimulusFile,
        replay_stimulus_file: ReplayStimulusFile,
        use_lowpass_filter: bool,
        zscore_threshold: float) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    sync_path: str
        The path to the sync file

    behavior_stimulus_file: BehaviorStimulusFile

    mapping_stimulus_file: MappingStimulusFile

    replay_stimulus_file: ReplayStimulusFile

    use_lowpass_filter: bool
        whther or not to apply a low pass filter to the
        running speed results

    zscore_threshold: float
        The threshold to use for removing outlier
        running speeds which might be noise and not true signal

    Returns
    -------
    A dict containing two data frames.
        'running_speed': A dataframe with mapping time to speed
        'running_acquisition': A dataframe mapping time to raw data
                               coming off the running wheel
    """
    (velocity_data,
     acq_data) = multi_stim_running_df_from_raw_data(
                    sync_path=sync_path,
                    behavior_stimulus_file=behavior_stimulus_file,
                    mapping_stimulus_file=mapping_stimulus_file,
                    replay_stimulus_file=replay_stimulus_file,
                    use_lowpass_filter=use_lowpass_filter,
                    zscore_threshold=zscore_threshold,
                    behavior_start_frame=0)

    running_speed = pd.DataFrame(
                      data={
                            'timestamps': velocity_data.frame_time.values,
                            'speed': velocity_data.velocity.values
                      })

    running_acq = pd.DataFrame(
                     data={
                         'dx': acq_data.dx.values,
                         'timestamps': acq_data.frame_time.values,
                         'v_in': acq_data.vin.values,
                         'v_sig': acq_data.vsig.values
                     }).set_index('timestamps')

    return {'running_speed': running_speed,
            'running_acquisition': running_acq}


def dual_stim_running_df_from_raw_data(
    sync_path: str,
    behavior_stimulus_file: BehaviorStimulusFile,
    mapping_stimulus_file: MappingStimulusFile,
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
    behavior_stimulus_file: BehaviorStimulusFile
        stimulus file for the behavior stimulus block
    mapping_stimulus_file: MappingStimulusFile
        stimulus file for the mapping stimulus block
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

    with sync_dataset.Dataset(sync_path) as sync_data:
        order = get_stim_order(sync_data, [behavior_stimulus_file, mapping_stimulus_file])

    if order[0] == 0:
        timestamp_results = get_stim_timestamps_from_stimulus_blocks(
                                  stimulus_files=[behavior_stimulus_file,
                                                  mapping_stimulus_file],
                                  sync_file=sync_path,
                                  raw_frame_time_lines=['frames',
                                                        'stim_vsync',
                                                        'vsync_stim'],
                                  raw_frame_time_direction='rising',
                                  frame_count_tolerance=0.0)

        behavior_timestamps = timestamp_results["timestamps"][0]
        mapping_timestamps = timestamp_results["timestamps"][1]
        start_frame = timestamp_results["start_frames"][0]
    else:
        timestamp_results = get_stim_timestamps_from_stimulus_blocks(
                                  stimulus_files=[mapping_stimulus_file,
                                                  behavior_stimulus_file],
                                  sync_file=sync_path,
                                  raw_frame_time_lines=['frames',
                                                        'stim_vsync',
                                                        'vsync_stim'],
                                  raw_frame_time_direction='rising',
                                  frame_count_tolerance=0.0)

        behavior_timestamps = timestamp_results["timestamps"][1]
        mapping_timestamps = timestamp_results["timestamps"][0]
        start_frame = timestamp_results["start_frames"][1]

    #replay_timestamps = timestamp_results["timestamps"][2]

    #start_frame = timestamp_results["start_frames"][0]

    behavior_velocities = _extract_dx_info(
        frame_times=behavior_timestamps,
        stimulus_file=behavior_stimulus_file,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    mapping_velocities = _extract_dx_info(
        frame_times=mapping_timestamps,
        stimulus_file=mapping_stimulus_file,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )

    """
    replay_velocities = _extract_dx_info(
        frame_times=replay_timestamps,
        stimulus_file=replay_stimulus_file,
        zscore_threshold=zscore_threshold,
        use_lowpass_filter=use_lowpass_filter
    )
    """
    all_frame_times = np.concatenate(
            [behavior_timestamps,
             mapping_timestamps])

    velocities, raw_data = _merge_dx_data_no_replay(
        mapping_velocities=mapping_velocities,
        behavior_velocities=behavior_velocities,
        frame_times=all_frame_times,
        behavior_start_frame=start_frame
    )

    return (velocities, raw_data)


def _get_dual_stim_running_df(
        sync_path: str,
        behavior_stimulus_file: BehaviorStimulusFile,
        mapping_stimulus_file: MappingStimulusFile,
        use_lowpass_filter: bool,
        zscore_threshold: float) -> Dict[str, pd.DataFrame]:
    """
    Parameters
    ----------
    sync_path: str
        The path to the sync file

    behavior_stimulus_file: BehaviorStimulusFile

    mapping_stimulus_file: MappingStimulusFile

    use_lowpass_filter: bool
        whther or not to apply a low pass filter to the
        running speed results

    zscore_threshold: float
        The threshold to use for removing outlier
        running speeds which might be noise and not true signal

    Returns
    -------
    A dict containing two data frames.
        'running_speed': A dataframe with mapping time to speed
        'running_acquisition': A dataframe mapping time to raw data
                               coming off the running wheel
    """
    (velocity_data,
     acq_data) = dual_stim_running_df_from_raw_data(
                    sync_path=sync_path,
                    behavior_stimulus_file=behavior_stimulus_file,
                    mapping_stimulus_file=mapping_stimulus_file,
                    use_lowpass_filter=use_lowpass_filter,
                    zscore_threshold=zscore_threshold,
                    behavior_start_frame=0)

    running_speed = pd.DataFrame(
                      data={
                            'timestamps': velocity_data.frame_time.values,
                            'speed': velocity_data.velocity.values
                      })

    running_acq = pd.DataFrame(
                     data={
                         'dx': acq_data.dx.values,
                         'timestamps': acq_data.frame_time.values,
                         'v_in': acq_data.vin.values,
                         'v_sig': acq_data.vsig.values
                     }).set_index('timestamps')

    return {'running_speed': running_speed,
            'running_acquisition': running_acq}
