"""
Created on Sunday July 15 2018

@author: marinag
"""
from itertools import chain
from typing import Dict, Any, Optional, List, Union
from allensdk.brain_observatory.behavior.sync.process_sync import (
    filter_digital, calculate_delay)  # NOQA: E402
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset  # NOQA: E402
import numpy as np
import scipy.stats as sps


def get_raw_stimulus_frames(
    dataset: SyncDataset, 
    permissive: bool = False
) -> np.ndarray:
    """ Report the raw timestamps of each stimulus frame. This corresponds to 
    the time at which the psychopy window's flip method returned, but not 
    necessarily to the time at which the stimulus frame was displayed.

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start).

    """
    try:
        return dataset.get_edges("falling",'stim_vsync', "seconds")
    except KeyError:
        if not permissive:
            raise
        return


def get_ophys_frames(
    dataset: SyncDataset, 
    permissive: bool = False
) -> np.ndarray:
    """ Report the timestamps of each optical physiology video frame

    Parameters
    ----------
    dataset : describes experiment timing

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start).
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Notes
    -----
    use rising edge for Scientifica, falling edge for Nikon 
    http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
    This function uses rising edges

    """
    try:
        return dataset.get_edges("rising", '2p_vsync', "seconds")
    except KeyError:
        if not permissive:
            raise
        return


def get_lick_times(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Report the timestamps of each detected lick

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no lick timestamps were found in this sync 
        dataset.

    """
    return dataset.get_edges(
        "rising", ["lick_times", "lick_sensor"], "seconds", permissive)
    

def get_stim_photodiode(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[List[float]]:
    """ Report the timestamps of each detected sync square transition (both 
    black -> white and white -> black) in this experiment.

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no photodiode timestamps were found in this sync 
        dataset.

    """
    return dataset.get_edges(
        "all", ["stim_photodiode", "photodiode"], "seconds", permissive)


def get_trigger(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Returns (as a 1-element array) the time at which optical physiology 
    acquisition was started.

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

   Returns
    -------
    timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no timestamps were found in this sync dataset.

    Notes
    -----
    Ophys frame timestamps can be recorded before acquisition start when 
        experimenters are setting up the recording session. These do not 
        correspond to acquired ophys frames.

    """
    return dataset.get_edges(
        "rising", ["2p_trigger", "acq_trigger"], "seconds", permissive)


def get_eye_tracking(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Report the timestamps of each frame of the eye tracking video

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no eye tracking timestamps were found in this sync 
        dataset.

    """
    return dataset.get_edges(
        "rising", ["cam2_exposure", "eye_tracking"], "seconds", permissive)


def get_behavior_monitoring(
    dataset: SyncDataset, 
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Report the timestamps of each frame of the behavior 
    monitoring video

    Parameters
    ----------
    dataset : describes experiment timing
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised

    Returns
    -------
    array of timestamps (floating point; seconds; relative to experiment start) 
        or None. If None, no behavior monitoring timestamps were found in this 
        sync dataset.

    """
    return dataset.get_edges(
        "rising", ["cam1_exposure", "behavior_monitoring"], "seconds", 
        permissive)


def get_sync_data(
    sync_path: str,
    permissive: bool = False
) -> Dict[str, Union[List, np.ndarray, None]]:
    """ Convenience function for extracting several timestamp arrays from a 
    sync file.

    Parameters
    ----------
    sync_path : The hdf5 file here ought to be a Visual Behavior sync output 
        file. See allensdk.brain_observatory.sync_dataset for more details of 
        this format.
    permissive : If True, None will be returned if timestamps are not found. If 
        False, a KeyError will be raised
    
    Returns
    -------
    A dictionary with the following keys. All timestamps in seconds:
        ophys_frames : timestamps of each optical physiology frame
        lick_times : timestamps of each detected lick
        ophys_trigger : The time at which ophys acquisition was started
        eye_tracking : timestamps of each eye tracking video frame
        behavior_monitoring : timestamps of behavior monitoring video frame
        stim_photodiode : timestamps of each photodiode transition
        stimulus_times_no_delay : raw stimulus frame timestamps
    Some values may be None. This indicates that the corresponding timestamps 
    were not located in this sync file.

    """

    sync_dataset = SyncDataset(sync_path)
    return {
        'ophys_frames': get_ophys_frames(sync_dataset, permissive),
        'lick_times': get_lick_times(sync_dataset, permissive),
        'ophys_trigger': get_trigger(sync_dataset, permissive),
        'eye_tracking': get_eye_tracking(sync_dataset, permissive),
        'behavior_monitoring': get_behavior_monitoring(sync_dataset, permissive),
        'stim_photodiode': get_stim_photodiode(sync_dataset, permissive),
        'stimulus_times_no_delay': get_raw_stimulus_frames(sync_dataset, permissive)
    }


def frame_time_offset(data: Dict[str, Any]) -> float:
    """
    Contained in the behavior "pickle" file is a series of time between
    consecutive vsync frames (`intervalsms`). This information required
    to get the timestamp (via frame number) for events that occured
    outside of a trial(e.g. licks). However, we don't have the value
    in the trial log time stream when the first vsync frame actually
    occured -- so we estimate it with a linear regression (frame
    number x time). All trials in the `trial_log` have events for
    `trial_start` and `trial_end`, so these are used to fit the
    regression. A linear regression is used rather than just
    subtracting the time from the first trial, since there can be some
    jitter given the 60Hz refresh rate.

    Parameters
    ----------
    data: dict
        behavior pickle well-known file data

    Returns
    -------
    float
        Time offset to add to the vsync stream to sync it with the
        `trial_log` time stream. The "zero-th" frame time.
    """
    events = [trial["events"] for trial
              in data["items"]["behavior"]["trial_log"]]
    # First event in `events` is `trial_start`, and last is `trial_end`
    # Event log has following schema:
    #    event name, description: 'enter'/'exit'/'', time, frame number
    # We want last two fields for first and last trial event
    trial_by_frame = list(chain(
        [event[i][-2:] for event in events for i in [0, -1]]))
    times = [trials[0] for trials in trial_by_frame]
    frames = [trials[1] for trials in trial_by_frame]
    time_to_first_vsync = sps.linregress(frames, times).intercept
    return time_to_first_vsync


def get_stimulus_rebase_function(data, stimulus_timestamps_no_monitor_delay):
    """
    Create a rebase function to align times for licks and stimulus timestamps
    in the "pickle" log with the same events in the event "sync" log.
    """
    vsyncs = data["items"]["behavior"]['intervalsms']
    stimulus_timestamps_pickle_pre = np.hstack((0, vsyncs)).cumsum() / 1000.0

    if (len(stimulus_timestamps_pickle_pre)
            != len(stimulus_timestamps_no_monitor_delay)):
        raise ValueError("Number of stimulus timestamps in pickle file are "
                         "not equal to timestamps in the sync file (pickle="
                         f"{len(stimulus_timestamps_pickle_pre)}, sync="
                         f"{len(stimulus_timestamps_no_monitor_delay)}).")
    time_to_first_vsync = frame_time_offset(data)
    stimulus_timestamps_pickle = (stimulus_timestamps_pickle_pre
                                  + time_to_first_vsync)
    # Rebase used to transform trial log times to sync times:
    time_slope, time_intercept, _, _, _ = sps.linregress(
        stimulus_timestamps_pickle, stimulus_timestamps_no_monitor_delay)

    # return transform of trial log time to sync time
    return lambda t: time_intercept + time_slope * t
